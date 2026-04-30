from typing import Any, cast

import pytest

from graphrag.query.structured_search.tog_search.pruning import LLMPruning


class _FakeResponseOutput:
    content = "[8, 3]"


class _FakeResponse:
    output = _FakeResponseOutput()


class _FakeChatModel:
    def __init__(self):
        self.prompts = []

    async def achat(self, prompt, history, model_parameters):
        self.prompts.append(prompt)
        return _FakeResponse()


@pytest.mark.asyncio
async def test_llm_relation_scoring_includes_path_and_history():
    model = _FakeChatModel()
    pruning = LLMPruning(model=cast(Any, model))  # noqa: TC006

    scored_relations, _ = await pruning.score_relations(
        query="Who founded the company?",
        entity_name="Company",
        relations=[
            ("organization.founders", "person", "outgoing", 1.0),
            ("organization.location", "place", "outgoing", 0.5),
        ],
        relation_history="Topic --[owns]--> Company",
        current_path="Topic --[owns]--> Company",
    )

    assert len(model.prompts) == 1
    prompt = model.prompts[0]
    assert "Who founded the company?" in prompt
    assert "Entity: Company" in prompt
    assert "Current reasoning path:" in prompt
    assert "Topic --[owns]--> Company" in prompt
    assert "Previous relations followed:" in prompt
    assert "organization.founders" in prompt
    assert scored_relations == [
        ("organization.founders", "person", "outgoing", 1.0, 8.0),
        ("organization.location", "place", "outgoing", 0.5, 3.0),
    ]


@pytest.mark.asyncio
async def test_llm_relation_scoring_keeps_custom_prompt_compatibility():
    model = _FakeChatModel()
    pruning = LLMPruning(
        model=cast(Any, model),  # noqa: TC006
        relation_scoring_prompt="Q={query}; E={entity_name}; R={relations}",
    )

    scored_relations, _ = await pruning.score_relations(
        query="question",
        entity_name="Entity",
        relations=[("relation", "target", "incoming", 1.0)],
        relation_history="Entity <--[previous]-- Topic",
        current_path="Entity <--[previous]-- Topic",
    )

    assert model.prompts == [
        "Q=question; E=Entity; R=1. [incoming] relation... (weight: 1.00)"
    ]
    assert scored_relations == [("relation", "target", "incoming", 1.0, 8.0)]
