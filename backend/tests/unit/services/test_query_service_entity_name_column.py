"""Unit tests for ToG entity-name column fallback behavior."""

from __future__ import annotations

import pandas as pd

from backend.app.services.query_service import _preferred_entity_name_column


def test_preferred_entity_name_column_uses_title_when_available() -> None:
    frame = pd.DataFrame({"title": ["A"], "id": ["1"]})
    assert _preferred_entity_name_column(frame) == "title"


def test_preferred_entity_name_column_falls_back_to_id() -> None:
    frame = pd.DataFrame({"id": ["1"], "description": ["x"]})
    assert _preferred_entity_name_column(frame) == "id"

