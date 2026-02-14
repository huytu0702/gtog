import sys
import asyncio
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _fake_cosmos_repo_init(self):
    self._collection_repo = MagicMock()
    self._collection_repo.get.return_value = None
    self._collection_repo.create.side_effect = lambda collection_id, description=None: SimpleNamespace(
        id=collection_id,
        name=collection_id,
        description=description,
        created_at=datetime.now(),
    )
    self._document_repo = MagicMock()
    self._document_repo.put.side_effect = lambda collection_id, filename, content_bytes: SimpleNamespace(
        collection_id=collection_id,
        name=filename,
        size=len(content_bytes),
        uploaded_at=datetime.now(),
    )
    self._prompt_repo = MagicMock()


def test_create_collection_cosmos_mode_does_not_create_local_workspace(tmp_path):
    """Cosmos mode collection creation should not materialize local input/output/cache directories."""
    with patch("app.services.storage_service.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.collections_dir = tmp_path

        with patch(
            "app.services.storage_service.StorageService._init_cosmos_repositories",
            new=_fake_cosmos_repo_init,
        ):
            from app.services.storage_service import StorageService

            svc = StorageService()
            svc.create_collection("demo")

            assert not (tmp_path / "demo" / "input").exists()
            assert not (tmp_path / "demo" / "output").exists()
            assert not (tmp_path / "demo" / "cache").exists()


def test_upload_document_cosmos_mode_writes_to_input_storage(tmp_path):
    """Cosmos mode upload should write document bytes into GraphRAG input storage."""
    async def run_test():
        with patch("app.services.storage_service.settings") as mock_settings:
            mock_settings.storage_mode = "cosmos"
            mock_settings.is_cosmos_mode = True
            mock_settings.collections_dir = tmp_path

            with patch(
                "app.services.storage_service.StorageService._init_cosmos_repositories",
                new=_fake_cosmos_repo_init,
            ):
                from app.services.storage_service import StorageService

                svc = StorageService()
                svc._collection_repo.get.return_value = SimpleNamespace(id="demo")

                mock_cfg = MagicMock()
                mock_cfg.input = MagicMock()
                mock_storage = MagicMock()
                mock_storage.set = AsyncMock()

                upload = UploadFile(filename="note.txt", file=BytesIO(b"hello cosmos"))

                with patch(
                    "app.services.storage_service.load_graphrag_config",
                    return_value=mock_cfg,
                    create=True,
                ):
                    with patch(
                        "app.services.storage_service.create_storage_from_config",
                        return_value=mock_storage,
                        create=True,
                    ):
                        await svc.upload_document("demo", upload)

                mock_storage.set.assert_awaited_once_with("note.txt", b"hello cosmos")

    asyncio.run(run_test())
