# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""LiteLLM usage telemetry helpers."""

import json
import os
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

RequestType = Literal["chat", "embedding"]

_usage_log_path: ContextVar[Path | None] = ContextVar("usage_log_path", default=None)
_usage_workflow: ContextVar[str | None] = ContextVar("usage_workflow", default=None)
_write_lock = threading.Lock()


@dataclass(frozen=True)
class TokenUsage:
    """Token usage normalized across chat and embedding responses."""

    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class UsageLogContext:
    """Data needed to write a single usage telemetry record."""

    model_provider: str
    model: str
    request_type: RequestType
    latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    success: bool
    cache_hit: bool | None
    cost_usd: float | None
    error_type: str | None = None
    error_message: str | None = None


def set_usage_log_path(
    path: str | Path | None,
    reset_token: Token[Path | None] | None = None,
) -> Token[Path | None] | None:
    """Set or reset the active usage log path."""
    if reset_token is not None:
        _usage_log_path.reset(reset_token)
        return None
    return _usage_log_path.set(Path(path) if path is not None else None)


def set_usage_workflow(
    workflow: str | None,
    reset_token: Token[str | None] | None = None,
) -> Token[str | None] | None:
    """Set or reset the active workflow label."""
    if reset_token is not None:
        _usage_workflow.reset(reset_token)
        return None
    return _usage_workflow.set(workflow)


def extract_token_usage(response: Any, request_type: RequestType) -> TokenUsage:
    """Extract token usage from a LiteLLM response."""
    usage = _get_value(response, "usage")
    if usage is None:
        return TokenUsage(prompt_tokens=None, output_tokens=None, total_tokens=None)

    prompt_tokens = _as_int(_get_value(usage, "prompt_tokens"))
    total_tokens = _as_int(_get_value(usage, "total_tokens"))
    output_tokens = 0 if request_type == "embedding" else _as_int(
        _get_value(usage, "completion_tokens")
    )

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def calculate_cost_usd(
    model_provider: str,
    model: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Calculate estimated cost from GRAPHRAG_LLM_PRICING_JSON."""
    if prompt_tokens is None and output_tokens is None:
        return None

    pricing_json = os.getenv("GRAPHRAG_LLM_PRICING_JSON")
    if not pricing_json:
        return None

    try:
        pricing = json.loads(pricing_json)
    except json.JSONDecodeError:
        return None

    model_key = f"{model_provider}/{model}"
    price = pricing.get(model_key) or pricing.get(model)
    if not isinstance(price, dict):
        return None

    prompt_price = _as_float(price.get("prompt_per_1m")) or 0.0
    completion_price = _as_float(price.get("completion_per_1m")) or 0.0

    return ((prompt_tokens or 0) * prompt_price + (output_tokens or 0) * completion_price) / 1_000_000


def write_usage_record(context: UsageLogContext) -> None:
    """Append a usage record to the active JSONL file if a path is configured."""
    try:
        path = _resolve_usage_path()
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "index" if _usage_workflow.get() else None,
            "workflow": _usage_workflow.get(),
            "request_type": context.request_type,
            "model_provider": context.model_provider,
            "model": context.model,
            "prompt_tokens": context.prompt_tokens,
            "output_tokens": context.output_tokens,
            "total_tokens": context.total_tokens,
            "latency_ms": context.latency_ms,
            "success": context.success,
            "cache_hit": context.cache_hit,
            "cost_usd": context.cost_usd,
            "error_type": context.error_type,
            "error_message": context.error_message,
        }

        with _write_lock, path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        return


def _resolve_usage_path() -> Path | None:
    context_path = _usage_log_path.get()
    if context_path is not None:
        return context_path

    env_path = os.getenv("GRAPHRAG_LLM_USAGE_PATH")
    if not env_path:
        return None

    path = Path(env_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return Path.cwd() / path


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
