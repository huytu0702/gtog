# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the Ragas simple-results runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from graphrag.eval.ragas_runner import (
    aggregate_ragas_results,
    build_ragas_scorers,
    main,
    parse_simple_results,
    resolve_settings_path,
)


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
            "scores": {
                "faithfulness": 0.5,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "answer_relevancy": 0.4,
                "answer_correctness": 0.25,
            },
        },
        {
            "status": "success",
            "method": "tog",
            "scores": {
                "faithfulness": 1.0,
                "context_precision": 0.5,
                "context_recall": 0.0,
                "answer_relevancy": 0.8,
                "answer_correctness": 0.75,
            },
        },
        {"status": "failed", "method": "local", "error": "boom"},
        {"status": "skipped", "method": "basic", "error": "missing"},
    ]

    summary = aggregate_ragas_results(
        detailed_results,
        model="gpt-5.2",
        base_url="http://127.0.0.1:8317/v1",
        input_path="eval_results_simple.json",
        settings_path="backend/settings.yaml",
    )

    assert summary["overall"] == {
        "count": 4,
        "success_count": 2,
        "fail_count": 1,
        "skip_count": 1,
    }
    assert summary["by_method"]["tog"]["count"] == 2
    assert summary["by_method"]["tog"]["faithfulness"] == 0.75
    assert summary["by_method"]["tog"]["context_precision"] == 0.75
    assert summary["by_method"]["tog"]["context_recall"] == 0.5
    assert summary["by_method"]["tog"]["answer_relevancy"] == 0.6
    assert summary["by_method"]["tog"]["answer_correctness"] == 0.5
    assert summary["by_method"]["local"]["fail_count"] == 1
    assert summary["by_method"]["basic"]["skip_count"] == 1
    assert summary["metadata"]["settings_path"].endswith("backend\\settings.yaml")


def test_resolve_settings_path_returns_resolved_path(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("models: {}\n", encoding="utf-8")

    assert resolve_settings_path(settings_path) == settings_path.resolve()


def test_build_ragas_scorers_uses_new_metric_keys(monkeypatch):
    fake_embeddings = object()
    fake_llm = object()

    class FakeMetric:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    metric_classes = {
        "ContextPrecision": FakeMetric,
        "ContextRecall": FakeMetric,
        "Faithfulness": FakeMetric,
        "AnswerRelevancy": FakeMetric,
        "AnswerCorrectness": FakeMetric,
        "AnswerSimilarity": FakeMetric,
    }

    monkeypatch.setattr(
        "graphrag.eval.ragas_runner.build_ragas_embeddings",
        lambda settings_path=None: fake_embeddings,
    )
    monkeypatch.setattr(
        "graphrag.eval.ragas_runner.build_ragas_llm",
        lambda **kwargs: fake_llm,
    )
    monkeypatch.setattr(
        "graphrag.eval.ragas_runner._import_ragas_metric",
        lambda name: metric_classes[name],
    )

    scorers = build_ragas_scorers(
        model="gpt-5.2",
        api_key="test-key",
        base_url="http://127.0.0.1:8317/v1",
        timeout=120.0,
        max_retries=5,
        settings_path="backend/settings.yaml",
    )

    assert set(scorers) == {
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    }
    assert scorers["context_precision"].kwargs["llm"] is fake_llm
    assert scorers["answer_relevancy"].kwargs["llm"] is fake_llm
    assert scorers["answer_relevancy"].kwargs["embeddings"] is fake_embeddings
    assert scorers["answer_correctness"].kwargs["llm"] is fake_llm
    assert scorers["answer_correctness"].kwargs["embeddings"] is fake_embeddings
    assert isinstance(scorers["answer_correctness"].kwargs["answer_similarity"], FakeMetric)
    assert (
        scorers["answer_correctness"].kwargs["answer_similarity"].kwargs["embeddings"]
        is fake_embeddings
    )


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
            "context_precision": FakeMetric("context_precision"),
            "context_recall": FakeMetric("context_recall"),
            "answer_relevancy": FakeMetric("answer_relevancy"),
            "answer_correctness": FakeMetric("answer_correctness"),
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
    assert detailed[1]["scores"]["context_precision"] == 1.0
    assert detailed[1]["scores"]["answer_relevancy"] == 1.0
    assert detailed[1]["scores"]["answer_correctness"] == 1.0
    assert detailed[2]["error"] == "proxy failure"
    assert summary["overall"]["success_count"] == 1
    assert summary["overall"]["fail_count"] == 1
    assert summary["overall"]["skip_count"] == 1
    assert summary["by_method"]["tog"]["context_recall"] == 1.0
