"""Tests for eval CLI module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_eval_cli_loads_config():
    """eval_cli should load eval_config.yaml."""
    from graphrag.cli.eval import eval_cli

    with patch("graphrag.cli.eval.EvalConfig.from_yaml") as mock_load:
        mock_config = MagicMock()
        mock_config.dataset.path = "eval/qa_eval.json"
        mock_config.indexes = {"tt1": "tt1"}
        mock_config.methods = ["tog"]
        mock_config.output.dir = "eval/results"
        mock_load.return_value = mock_config

        with patch("graphrag.cli.eval.run_evaluation") as mock_run:
            mock_run.return_value = None

            # Should not raise
            eval_cli(
                root_dir=Path("."),
                eval_config=Path("eval_config.yaml"),
                methods=None,
                resume=False,
                skip_evaluation=False,
                verbose=False,
            )

        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_run_evaluation_skip_mode_preserves_efficiency_for_simple_output(tmp_path):
    """run_evaluation should preserve efficiency metrics when creating simple output."""
    from graphrag.cli.eval import run_evaluation
    from graphrag.eval.runner import EfficiencyMetrics, QueryResult

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "Who?",
                    "ground_truth": "Answer",
                    "context": "Some context",
                }
            ]
        )
    )

    eval_cfg = MagicMock()
    eval_cfg.dataset.path = "dataset.json"
    eval_cfg.indexes = {"default": "unused"}
    eval_cfg.methods = ["tog"]
    eval_cfg.output.dir = str(tmp_path / "eval_results")
    eval_cfg.output.save_intermediate = False

    runner_result = QueryResult(
        question="Who?",
        method="tog",
        response="Answer",
        context="Some context",
        context_text="Search context",
        ground_truth="Answer",
        efficiency=EfficiencyMetrics(
            latency=1.5,
            llm_calls=3,
            prompt_tokens=120,
            output_tokens=40,
        ),
    )

    mock_runner = MagicMock()
    mock_runner._load_index = AsyncMock(return_value={})
    mock_runner.run_evaluation = AsyncMock(return_value=[runner_result])

    captured: dict = {}

    class FakeAggregated:
        by_method = {}

        def save_simple(self, output_dir: str) -> None:
            captured["saved_output_dir"] = output_dir

        def save(self, output_dir: str) -> None:
            captured["saved_output_dir"] = output_dir

    def aggregate_side_effect(query_results):
        captured["query_results"] = query_results
        return FakeAggregated()

    with (
        patch("graphrag.cli.eval.load_config", return_value=MagicMock()),
        patch("graphrag.cli.eval.EvaluationRunner", return_value=mock_runner),
        patch("graphrag.cli.eval.aggregate_results", side_effect=aggregate_side_effect),
    ):
        await run_evaluation(
            root=tmp_path,
            eval_cfg=eval_cfg,
            resume=False,
            skip_evaluation=True,
            verbose=False,
        )

    query_result = captured["query_results"][0]
    assert query_result.efficiency is not None
    assert query_result.efficiency.latency == 1.5
    assert query_result.efficiency.llm_calls == 3
    assert query_result.efficiency.prompt_tokens == 120
    assert query_result.efficiency.output_tokens == 40
