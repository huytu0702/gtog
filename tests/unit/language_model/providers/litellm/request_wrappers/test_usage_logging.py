import json
from pathlib import Path
from types import SimpleNamespace

from graphrag.language_model.providers.litellm.request_wrappers.usage_logging import (
    UsageLogContext,
    calculate_cost_usd,
    extract_token_usage,
    set_usage_log_path,
    set_usage_workflow,
    write_usage_record,
)


def test_extract_token_usage_from_chat_response():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
        )
    )

    usage = extract_token_usage(response, "chat")

    assert usage.prompt_tokens == 100
    assert usage.output_tokens == 25
    assert usage.total_tokens == 125


def test_extract_token_usage_from_embedding_response():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=77,
            total_tokens=77,
        )
    )

    usage = extract_token_usage(response, "embedding")

    assert usage.prompt_tokens == 77
    assert usage.output_tokens == 0
    assert usage.total_tokens == 77


def test_calculate_cost_uses_model_pricing(monkeypatch):
    monkeypatch.setenv(
        "GRAPHRAG_LLM_PRICING_JSON",
        json.dumps({
            "openai/gpt-test": {
                "prompt_per_1m": 2.0,
                "completion_per_1m": 4.0,
            }
        }),
    )

    cost = calculate_cost_usd("openai", "gpt-test", 1_000_000, 500_000)

    assert cost == 4.0


def test_write_usage_record_writes_jsonl(tmp_path: Path):
    output_path = tmp_path / "llm_usage.jsonl"
    path_token = set_usage_log_path(output_path)
    workflow_token = set_usage_workflow("extract_graph")

    try:
        write_usage_record(
            UsageLogContext(
                model_provider="openai",
                model="gpt-test",
                request_type="chat",
                latency_ms=12.5,
                prompt_tokens=10,
                output_tokens=5,
                total_tokens=15,
                success=True,
                cache_hit=False,
                cost_usd=None,
            )
        )
    finally:
        set_usage_workflow(None, workflow_token)
        set_usage_log_path(None, path_token)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["phase"] == "index"
    assert rows[0]["workflow"] == "extract_graph"
    assert rows[0]["request_type"] == "chat"
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["output_tokens"] == 5


def test_write_usage_record_is_best_effort(tmp_path: Path):
    path_token = set_usage_log_path(tmp_path)

    try:
        write_usage_record(
            UsageLogContext(
                model_provider="openai",
                model="gpt-test",
                request_type="chat",
                latency_ms=12.5,
                prompt_tokens=10,
                output_tokens=5,
                total_tokens=15,
                success=True,
                cache_hit=False,
                cost_usd=None,
            )
        )
    finally:
        set_usage_log_path(None, path_token)
