"""Tests for document upload size limits."""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from app.routers import documents as documents_router


def test_upload_size_limit_exceeded():
    """Oversized uploads should raise HTTP 413."""
    file = UploadFile(
        filename="big.txt",
        file=BytesIO(b"a" * (documents_router.MAX_UPLOAD_BYTES + 1)),
    )

    with pytest.raises(HTTPException) as exc:
        documents_router._enforce_upload_size(file)

    assert exc.value.status_code == 413


def test_upload_size_limit_allows_equal_limit():
    """Uploads at the limit should be allowed."""
    file = UploadFile(
        filename="small.txt",
        file=BytesIO(b"a" * documents_router.MAX_UPLOAD_BYTES),
    )

    documents_router._enforce_upload_size(file)
