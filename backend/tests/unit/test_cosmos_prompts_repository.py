import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from app.repositories.cosmos_prompts import CosmosPromptRepository


@pytest.fixture
def mock_cosmos_container():
    """Mock Cosmos DB container."""
    container = MagicMock()
    container.upsert_item = MagicMock(return_value={})
    container.read_item = MagicMock(side_effect=Exception("Not found"))
    container.query_items = MagicMock(return_value=[])
    return container


def test_set_and_get_prompt(mock_cosmos_container):
    repo = CosmosPromptRepository(mock_cosmos_container)
    repo.set_prompt("demo", "local_search_system_prompt.txt", "prompt-text")
    
    # Verify upsert was called
    mock_cosmos_container.upsert_item.assert_called_once()
    call_args = mock_cosmos_container.upsert_item.call_args[0][0]
    assert call_args["content"] == "prompt-text"
    assert call_args["prompt_name"] == "local_search_system_prompt.txt"


def test_get_prompt_reads_from_cosmos(mock_cosmos_container):
    mock_cosmos_container.read_item = MagicMock(return_value={
        "id": "prompt:demo:local_search_system_prompt.txt",
        "kind": "prompt",
        "collection_id": "demo",
        "prompt_name": "local_search_system_prompt.txt",
        "content": "stored-prompt-text",
    })
    repo = CosmosPromptRepository(mock_cosmos_container)
    value = repo.get_prompt("demo", "local_search_system_prompt.txt")
    assert value == "stored-prompt-text"


def test_get_prompt_returns_none_when_missing(mock_cosmos_container):
    repo = CosmosPromptRepository(mock_cosmos_container)
    value = repo.get_prompt("demo", "missing.txt")
    assert value is None


def test_list_prompt_names_returns_names(mock_cosmos_container):
    mock_cosmos_container.query_items = MagicMock(return_value=[
        {"id": "prompt:demo:extract_graph.txt", "prompt_name": "extract_graph.txt"},
        {"id": "prompt:demo:local_search_system_prompt.txt", "prompt_name": "local_search_system_prompt.txt"},
    ])
    repo = CosmosPromptRepository(mock_cosmos_container)
    names = repo.list_prompt_names("demo")
    assert len(names) == 2
    assert "extract_graph.txt" in names
    assert "local_search_system_prompt.txt" in names


@patch("app.repositories.cosmos_prompts.load_default_prompt_texts")
def test_seed_defaults_writes_required_prompt_keys(mock_load_defaults, mock_cosmos_container):
    mock_load_defaults.return_value = {
        "extract_graph.txt": "graph prompt",
        "local_search_system_prompt.txt": "local prompt",
    }
    repo = CosmosPromptRepository(mock_cosmos_container)
    repo.seed_defaults("demo")
    
    # Verify upsert was called for each prompt
    assert mock_cosmos_container.upsert_item.call_count == 2
    
    # Check that the prompts were stored
    call_args_list = mock_cosmos_container.upsert_item.call_args_list
    prompt_names = [call[0][0]["prompt_name"] for call in call_args_list]
    assert "extract_graph.txt" in prompt_names
    assert "local_search_system_prompt.txt" in prompt_names
