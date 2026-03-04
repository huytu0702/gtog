"""Unit tests for serving repository value normalization."""

from __future__ import annotations

import numpy as np

from backend.app.repositories.serving_repository import (
    _normalize_row,
    _normalize_value,
    _source_id_from_record,
)


def test_normalize_value_handles_numpy_arrays() -> None:
    value = np.array([0.1, 0.2, 0.3], dtype=float)
    normalized = _normalize_value(value)
    assert normalized == [0.1, 0.2, 0.3]


def test_normalize_row_handles_embedding_like_values() -> None:
    row = {"id": "x1", "embedding": np.array([1, 2, 3], dtype=int)}
    normalized = _normalize_row(row)
    assert normalized["id"] == "x1"
    assert normalized["embedding"] == [1, 2, 3]


def test_source_id_from_record_does_not_boolean_eval_arrays() -> None:
    record = {"id": np.array([10, 20], dtype=int)}
    source_id = _source_id_from_record(record, 7)
    assert source_id == "[10, 20]"

