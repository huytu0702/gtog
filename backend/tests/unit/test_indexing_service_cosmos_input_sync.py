import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_cosmos_indexing_does_not_create_local_input_directory(tmp_path):
    """Cosmos mode indexing should not materialize documents to local input dir."""
    from app.models import IndexStatus, IndexStatusResponse
    from app.services.indexing_service import IndexingService

    async def run_test():
        with patch("app.services.indexing_service.settings") as mock_settings:
            mock_settings.is_cosmos_mode = True
            mock_settings.collections_dir = tmp_path

            with patch("app.services.indexing_service.load_graphrag_config") as mock_load_config:
                mock_load_config.return_value = MagicMock()

                with patch(
                    "app.services.indexing_service.api.build_index",
                    new_callable=AsyncMock,
                ) as mock_build_index:
                    mock_build_index.return_value = [MagicMock(errors=[])]

                    svc = IndexingService()
                    svc._get_document_repo = MagicMock(return_value=MagicMock(list=MagicMock(return_value=[])))
                    svc.indexing_tasks["demo"] = IndexStatusResponse(
                        collection_id="demo",
                        status=IndexStatus.RUNNING,
                    )

                    await svc._run_indexing_task("demo")

                    assert not (tmp_path / "demo" / "input").exists()

    asyncio.run(run_test())


def test_cosmos_indexing_runs_without_local_workspace_sync(tmp_path):
    """Cosmos mode should call GraphRAG indexing directly without local workspace sync."""
    from app.models import IndexStatus, IndexStatusResponse
    from app.services.indexing_service import IndexingService

    async def run_test():
        with patch("app.services.indexing_service.settings") as mock_settings:
            mock_settings.is_cosmos_mode = True
            mock_settings.collections_dir = tmp_path

            with patch("app.services.indexing_service.load_graphrag_config") as mock_load_config:
                mock_load_config.return_value = MagicMock()

                with patch(
                    "app.services.indexing_service.api.build_index",
                    new_callable=AsyncMock,
                ) as mock_build_index:
                    mock_build_index.return_value = [MagicMock(errors=[])]

                    svc = IndexingService()
                    svc._get_document_repo = MagicMock(return_value=MagicMock(list=MagicMock(return_value=[])))
                    svc.indexing_tasks["demo"] = IndexStatusResponse(
                        collection_id="demo",
                        status=IndexStatus.RUNNING,
                    )

                    await svc._run_indexing_task("demo")

                    mock_build_index.assert_awaited_once()

    asyncio.run(run_test())
