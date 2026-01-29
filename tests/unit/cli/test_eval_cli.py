"""Tests for eval CLI module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path


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
                imdb_key=None,
                resume=False,
                verbose=False,
            )

        mock_load.assert_called_once()
