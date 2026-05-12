import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphrag.language_model.providers.litellm.request_wrappers.usage_logging import (
    set_usage_log_path,
    set_usage_workflow,
)
from graphrag.language_model.providers.litellm.request_wrappers.with_logging import (
    with_logging,
)


@pytest.fixture
def model_config():
    return SimpleNamespace(model_provider="openai", model="gpt-test", deployment_name=None)


@pytest.fixture
def usage_context(tmp_path: Path):
    output_path = tmp_path / "llm_usage.jsonl"
    path_token = set_usage_log_path(output_path)
    workflow_token = set_usage_workflow("extract_graph")
    try:
        yield output_path
    finally:
        set_usage_workflow(None, workflow_token)
        set_usage_log_path(None, path_token)


def test_with_logging_records_sync_success(model_config, usage_context: Path):
    def sync_fn(**kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            )
        )

    async def async_fn(**kwargs):
        pytest.fail("async not used")

    wrapped_sync, _ = with_logging(
        sync_fn=sync_fn,
        async_fn=async_fn,
        model_config=model_config,
        request_type="chat",
    )

    wrapped_sync(messages=[])

    rows = [json.loads(line) for line in usage_context.read_text().splitlines()]
    assert rows[0]["success"] is True
    assert rows[0]["workflow"] == "extract_graph"
    assert rows[0]["prompt_tokens"] == 100
    assert rows[0]["output_tokens"] == 20


@pytest.mark.asyncio
async def test_with_logging_records_async_error(model_config, usage_context: Path):
    def sync_fn(**kwargs):
        pytest.fail("sync not used")

    async def async_fn(**kwargs):
        message = "provider failed"
        raise ValueError(message)

    _, wrapped_async = with_logging(
        sync_fn=sync_fn,
        async_fn=async_fn,
        model_config=model_config,
        request_type="embedding",
    )

    with pytest.raises(ValueError, match="provider failed"):
        await wrapped_async(input=["hello"])

    rows = [json.loads(line) for line in usage_context.read_text().splitlines()]
    assert rows[0]["success"] is False
    assert rows[0]["request_type"] == "embedding"
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["error_message"] is None
    assert rows[0]["prompt_tokens"] is None
