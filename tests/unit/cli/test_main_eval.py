"""Tests for eval command in main CLI."""

import pytest
from typer.testing import CliRunner
from graphrag.cli.main import app

runner = CliRunner()


def test_eval_command_exists():
    """CLI should have eval command."""
    commands = [c.name for c in app.registered_commands]
    print(f"Commands: {commands}")
    assert "eval" in commands
