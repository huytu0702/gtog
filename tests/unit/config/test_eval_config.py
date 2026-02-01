"""Tests for eval_config module."""

import pytest

from graphrag.config.models.eval_config import EvalConfig


def test_eval_config_from_dict():
    """EvalConfig should load from dictionary."""
    config_dict = {
        "dataset": {"path": "eval/qa_eval.json"},
        "indexes": {
            "tt0097576": "tt0097576",
            "tt0102798": "tt0102798",
        },
        "methods": ["tog", "local", "basic"],
        "output": {"dir": "eval/results", "save_intermediate": True},
        "judge": {"temperature": 0.0},
    }

    config = EvalConfig(**config_dict)

    assert config.dataset.path == "eval/qa_eval.json"
    assert len(config.indexes) == 2
    assert "tog" in config.methods


def test_eval_config_defaults():
    """EvalConfig should have sensible defaults."""
    config = EvalConfig(
        dataset={"path": "eval/qa_eval.json"},
        indexes={"tt1": "tt1"},
    )

    assert config.methods == ["tog", "local", "basic"]
    assert config.output.save_intermediate is True
