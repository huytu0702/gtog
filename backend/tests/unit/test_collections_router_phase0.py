import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.app.models import CollectionCreate
import backend.app.routers.collections as collections_router


def test_create_collection_has_single_valueerror_and_exception_handler():
    source = inspect.getsource(collections_router.create_collection)
    assert source.count("except ValueError as e:") == 1
    assert source.count("except Exception as e:") == 1


@pytest.mark.asyncio
async def test_create_collection_value_error_response_shape_is_stable():
    with patch(
        "backend.app.routers.collections.storage_service.create_collection",
        side_effect=ValueError("duplicate collection"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await collections_router.create_collection(
                request=MagicMock(),
                collection=CollectionCreate(name="valid_name", description="desc"),
            )

    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail
    assert detail["error"] == "Validation failed"
    assert detail["field"] == "name"
