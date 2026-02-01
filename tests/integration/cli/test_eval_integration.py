"""Tests for eval CLI integration."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from graphrag.cli.main import app

runner = CliRunner()


def test_eval_requires_config():
    """Eval should fail gracefully without config."""
    result = runner.invoke(app, ["eval", "--root", "/nonexistent"])
    # Should exit with error about missing config
    assert result.exit_code != 0 or "Error" in result.stdout
