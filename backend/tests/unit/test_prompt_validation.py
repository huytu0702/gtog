from pathlib import Path

import pytest

from backend.app.utils.helpers import _REQUIRED_PROMPT_FILES, _validate_shared_prompt_files


def test_validate_shared_prompt_files_raises_when_missing(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing required prompt files"):
        _validate_shared_prompt_files(prompt_dir)


def test_validate_shared_prompt_files_passes_when_complete(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    for filename in _REQUIRED_PROMPT_FILES:
        (prompt_dir / filename).write_text("placeholder", encoding="utf-8")

    _validate_shared_prompt_files(prompt_dir)
