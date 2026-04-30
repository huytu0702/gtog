from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from graphrag.query.structured_search.drift_search.primer import (
    DRIFTPrimer,
    _parse_primer_response,
)


class _TokenizerStub:
    def encode(self, text: str):
        return list(text)


class _ChatModelStub:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.config = SimpleNamespace(model="stub")

    async def achat(self, prompt: str, history: list | None = None, **kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(
            output=SimpleNamespace(content=self.responses.pop(0)),
            parsed_response=None,
        )


def test_split_reports_preserves_dataframe_columns():
    reports = pd.DataFrame([
        {"short_id": "0", "community_id": "0", "full_content": "Report A"},
        {"short_id": "1", "community_id": "1", "full_content": "Report B"},
    ])
    primer = DRIFTPrimer.__new__(DRIFTPrimer)
    primer.config = cast(Any, SimpleNamespace(primer_folds=2))

    folds = primer.split_reports(reports)

    assert len(folds) == 2
    assert [list(fold.columns) for fold in folds] == [
        ["short_id", "community_id", "full_content"],
        ["short_id", "community_id", "full_content"],
    ]
    assert [fold.iloc[0]["full_content"] for fold in folds] == ["Report A", "Report B"]


def test_parse_primer_response_repairs_fenced_json():
    parsed = _parse_primer_response(
        "```json\n{\n"
        '  "intermediate_answer": "# Answer\\n\\nSummary",\n'
        '  "score": "84",\n'
        '  "follow_up_queries": ["Q1", "Q2", "Q3", "Q4", "Q5"]\n'
        "}\n```"
    )

    assert parsed == {
        "intermediate_answer": "# Answer\n\nSummary",
        "score": 84,
        "follow_up_queries": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    }


def test_parse_primer_response_coerces_multiline_follow_ups():
    parsed = _parse_primer_response(
        '{"intermediate_answer": "# Answer", "score": 60, '
        '"follow_up_queries": "- one\\n- two\\n- three\\n- four\\n- five"}'
    )

    assert parsed["follow_up_queries"] == ["one", "two", "three", "four", "five"]


@pytest.mark.asyncio
async def test_decompose_query_uses_tolerant_json_parsing():
    primer = DRIFTPrimer(
        config=cast(Any, SimpleNamespace(primer_folds=1)),
        chat_model=cast(
            Any,
            _ChatModelStub(
            [
                "```json\n{\n"
                '  "intermediate_answer": "# Topic\\n\\nRecovered answer",\n'
                '  "score": 91,\n'
                '  "follow_up_queries": ["A", "B", "C", "D", "E"]\n'
                "}\n```"
            ]
            ),
        ),
        tokenizer=cast(Any, _TokenizerStub()),
    )
    reports = pd.DataFrame([{"full_content": "Report 1"}])

    parsed, token_ct = await primer.decompose_query("What is this about?", reports)

    assert parsed["score"] == 91
    assert parsed["follow_up_queries"] == ["A", "B", "C", "D", "E"]
    assert parsed["intermediate_answer"].startswith("# Topic")
    assert token_ct["llm_calls"] == 1


def test_parse_primer_response_rejects_missing_follow_up_queries():
    with pytest.raises(ValueError, match="follow_up_queries"):
        _parse_primer_response(
            '{"intermediate_answer": "# Answer", "score": 42, "follow_up_queries": []}'
        )
