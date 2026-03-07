# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the Ragas simple-results runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from graphrag.eval.ragas_runner import (
    aggregate_ragas_results,
    main,
    parse_simple_results,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_simple_results_skips_missing_required_fields():
    rows = [
        {
            "question": "Q1",
            "response": "A1",
            "ground_truth": "G1",
            "context_text": "C1",
            "method": "tog",
        },
        {
            "question": "",
            "response": "A2",
            "ground_truth": "G2",
            "context_text": "C2",
            "method": "tog",
        },
        {
            "question": "Q3",
            "response": "A3",
            "ground_truth": "G3",
            "context_text": "   ",
        },
    ]

    valid_rows, skipped_rows = parse_simple_results(rows)

    assert len(valid_rows) == 1
    assert valid_rows[0].to_sample_payload() == {
        "user_input": "Q1",
        "response": "A1",
        "reference": "G1",
        "retrieved_contexts": ["C1"],
    }
    assert len(skipped_rows) == 2
    assert skipped_rows[0]["status"] == "skipped"
    assert "question" in skipped_rows[0]["error"]
    assert "context_text" in skipped_rows[1]["error"]


def test_aggregate_ragas_results_groups_scores_by_method():
    detailed_results = [
        {
            "status": "success",
            "method": "tog",
            "scores": {"faithfulness": 0.5, "context_recall": 1.0},
        },
        {
            "status": "success",
            "method": "tog",
            "scores": {"faithfulness": 1.0, "context_recall": 0.0},
        },
        {"status": "failed", "method": "local", "error": "boom"},
        {"status": "skipped", "method": "basic", "error": "missing"},
    ]

    summary = aggregate_ragas_results(
        detailed_results,
        model="gpt-5.2",
        base_url="http://127.0.0.1:8317/v1",
        input_path="eval_results_simple.json",
    )

    assert summary["overall"] == {
        "count": 4,
        "success_count": 2,
        "fail_count": 1,
        "skip_count": 1,
    }
    assert summary["by_method"]["tog"]["count"] == 2
    assert summary["by_method"]["tog"]["faithfulness"] == 0.75
    assert summary["by_method"]["tog"]["context_recall"] == 0.5
    assert summary["by_method"]["local"]["fail_count"] == 1
    assert summary["by_method"]["basic"]["skip_count"] == 1


def test_main_writes_outputs_and_continues_after_row_failure(
    monkeypatch,
    tmp_path: Path,
):
    input_path = tmp_path / "eval_results_simple.json"
    output_dir = tmp_path / "ragas_out"
    input_path.write_text(
        json.dumps(
            [
                {
                    "question": "good",
                    "response": "A1",
                    "ground_truth": "G1",
                    "context_text": "C1",
                    "method": "tog",
                },
                {
                    "question": "bad",
                    "response": "A2",
                    "ground_truth": "G2",
                    "context_text": "C2",
                    "method": "tog",
                },
                {
                    "question": "skip",
                    "response": "",
                    "ground_truth": "G3",
                    "context_text": "C3",
                    "method": "basic",
                },
            ]
        ),
        encoding="utf-8",
    )

    class FakeMetric:
        def __init__(self, name: str):
            self.name = name

        def score(self, **kwargs) -> float:
            if kwargs["user_input"] == "bad":
                msg = "proxy failure"
                raise RuntimeError(msg)
            return 0.5 if self.name == "faithfulness" else 1.0

    monkeypatch.setattr(
        "graphrag.eval.ragas_runner.build_openai_client",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "graphrag.eval.ragas_runner.build_ragas_scorers",
        lambda **kwargs: {
            "faithfulness": FakeMetric("faithfulness"),
            "context_recall": FakeMetric("context_recall"),
        },
    )
    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    detailed = json.loads(
        (output_dir / "eval_results_ragas_detailed.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (output_dir / "eval_results_ragas_summary.json").read_text(encoding="utf-8")
    )

    assert [row["status"] for row in detailed] == ["skipped", "success", "failed"]
    assert detailed[1]["scores"]["faithfulness"] == 0.5
    assert detailed[2]["error"] == "proxy failure"
    assert summary["overall"]["success_count"] == 1
    assert summary["overall"]["fail_count"] == 1
    assert summary["overall"]["skip_count"] == 1
    assert summary["by_method"]["tog"]["context_recall"] == 1.0
