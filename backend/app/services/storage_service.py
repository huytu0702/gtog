"""Storage management service with file and Cosmos DB backends."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import UploadFile

from ..config import settings
from ..models import CollectionResponse, DocumentResponse


class StorageService:
    """Service for managing storage operations with file or Cosmos DB backend."""

    def __init__(self):
        """Initialize the storage service based on storage mode."""
        self.collections_dir = settings.collections_dir
        self._is_cosmos_mode = settings.is_cosmos_mode

        # Initialize repositories based on storage mode
        if self._is_cosmos_mode:
            self._init_cosmos_repositories()
        else:
            # Ensure collections directory exists for file mode
            self.collections_dir.mkdir(parents=True, exist_ok=True)

    def _init_cosmos_repositories(self):
        """Initialize Cosmos DB repositories."""
        try:
            from azure.cosmos import CosmosClient, PartitionKey

            self._cosmos_client = CosmosClient(
                url=settings.cosmos_endpoint,
                credential=settings.cosmos_key,
                connection_verify=False,
            )

            # Create database if it doesn't exist
            try:
                database = self._cosmos_client.create_database_if_not_exists(
                    id=settings.cosmos_database
                )
            except Exception as db_err:
                import logging
                logging.warning(f"Database may already exist or error creating: {db_err}")
                database = self._cosmos_client.get_database_client(settings.cosmos_database)

            # Create container if it doesn't exist
            try:
                container = database.create_container_if_not_exists(
                    id=settings.cosmos_container,
                    partition_key=PartitionKey(path="/id"),
                    offer_throughput=400
                )
            except Exception as container_err:
                import logging
                logging.warning(f"Container may already exist or error creating: {container_err}")
                container = database.get_container_client(settings.cosmos_container)

            from ..repositories.cosmos_collections import CosmosCollectionRepository
            from ..repositories.cosmos_documents import CosmosDocumentRepository
            from ..repositories.cosmos_prompts import CosmosPromptRepository

            self._collection_repo = CosmosCollectionRepository(container)
            self._document_repo = CosmosDocumentRepository(container)
            self._prompt_repo = CosmosPromptRepository(container)
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize Cosmos repositories: {e}")
            raise

    def _seed_default_prompts_file(self, prompts_dir: Path) -> None:
        """Seed default prompts to local filesystem."""
        from ..repositories.default_prompts import load_default_prompt_texts

        prompts_dir.mkdir(parents=True, exist_ok=True)
        default_prompts = load_default_prompt_texts()

        for prompt_name, content in default_prompts.items():
            (prompts_dir / prompt_name).write_text(content, encoding="utf-8")

    def create_collection(
        self, collection_id: str, description: Optional[str] = None
    ) -> CollectionResponse:
        """Create a new collection.

        Args:
            collection_id: Unique identifier for the collection
            description: Optional description of the collection

        Returns:
            CollectionResponse with collection details

        Raises:
            ValueError: If collection already exists
        """
        if self._is_cosmos_mode:
            # Check if collection already exists
            existing = self._collection_repo.get(collection_id)
            if existing:
                raise ValueError(f"Collection '{collection_id}' already exists")

            # Create collection in Cosmos
            record = self._collection_repo.create(collection_id, description)

            # Seed default prompts
            self._prompt_repo.seed_defaults(collection_id)

            return CollectionResponse(
                id=record.id,
                name=record.name,
                description=record.description,
                created_at=record.created_at,
                document_count=0,
                indexed=False,
            )
        else:
            # File mode: create local directory structure
            collection_dir = self.collections_dir / collection_id

            if collection_dir.exists():
                raise ValueError(f"Collection '{collection_id}' already exists")

            # Create directory structure (matching graphrag init)
            collection_dir.mkdir(parents=True)
            (collection_dir / "input").mkdir()
            (collection_dir / "output").mkdir()
            (collection_dir / "cache").mkdir()
            (collection_dir / "prompts").mkdir()

            # Generate default prompts from graphrag package
            try:
                self._seed_default_prompts_file(collection_dir / "prompts")
            except Exception as e:
                import logging
                logging.warning(
                    f"Failed to generate prompts for collection {collection_id}: {e}"
                )

            return CollectionResponse(
                id=collection_id,
                name=collection_id,
                description=description,
                created_at=datetime.now(),
                document_count=0,
                indexed=False,
            )

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its contents.

        Args:
            collection_id: The collection identifier

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If collection does not exist
        """
        if self._is_cosmos_mode:
            # Delete collection from Cosmos
            if not self._collection_repo.get(collection_id):
                raise ValueError(f"Collection '{collection_id}' not found")

            # Delete all documents in the collection
            docs = self._document_repo.list(collection_id)
            for doc in docs:
                self._document_repo.delete(collection_id, doc.name)

            # Delete the collection
            self._collection_repo.delete(collection_id)
            return True
        else:
            # File mode: delete local directory
            collection_dir = self.collections_dir / collection_id

            if not collection_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")

            shutil.rmtree(collection_dir)
            return True

    def list_collections(self) -> List[CollectionResponse]:
        """List all available collections.

        Returns:
            List of CollectionResponse objects
        """
        if self._is_cosmos_mode:
            records = self._collection_repo.list()
            return [
                CollectionResponse(
                    id=rec.id,
                    name=rec.name,
                    description=rec.description,
                    created_at=rec.created_at,
                    document_count=len(self._document_repo.list(rec.id)),
                    indexed=False,  # TODO: Check indexed status from Cosmos
                )
                for rec in records
            ]
        else:
            collections = []

            if not self.collections_dir.exists():
                return collections

            for collection_dir in self.collections_dir.iterdir():
                if collection_dir.is_dir():
                    collection_id = collection_dir.name
                    input_dir = collection_dir / "input"
                    output_dir = collection_dir / "output"

                    # Count documents
                    document_count = 0
                    if input_dir.exists():
                        document_count = len([
                            f for f in input_dir.iterdir() if f.is_file()
                        ])

                    # Check if indexed (has output files)
                    indexed = False
                    if output_dir.exists():
                        required_files = ["entities.parquet", "communities.parquet"]
                        indexed = all((output_dir / f).exists() for f in required_files)

                    collections.append(
                        CollectionResponse(
                            id=collection_id,
                            name=collection_id,
                            description=None,
                            created_at=datetime.fromtimestamp(
                                collection_dir.stat().st_ctime
                            ),
                            document_count=document_count,
                            indexed=indexed,
                        )
                    )

            return collections

    def get_collection(self, collection_id: str) -> Optional[CollectionResponse]:
        """Get details about a specific collection.

        Args:
            collection_id: The collection identifier

        Returns:
            CollectionResponse or None if not found
        """
        if self._is_cosmos_mode:
            record = self._collection_repo.get(collection_id)
            if not record:
                return None

            return CollectionResponse(
                id=record.id,
                name=record.name,
                description=record.description,
                created_at=record.created_at,
                document_count=len(self._document_repo.list(collection_id)),
                indexed=False,  # TODO: Check indexed status from Cosmos
            )
        else:
            collection_dir = self.collections_dir / collection_id

            if not collection_dir.exists():
                return None

            input_dir = collection_dir / "input"
            output_dir = collection_dir / "output"

            # Count documents
            document_count = 0
            if input_dir.exists():
                document_count = len([f for f in input_dir.iterdir() if f.is_file()])

            # Check if indexed
            indexed = False
            if output_dir.exists():
                required_files = ["entities.parquet", "communities.parquet"]
                indexed = all((output_dir / f).exists() for f in required_files)

            return CollectionResponse(
                id=collection_id,
                name=collection_id,
                description=None,
                created_at=datetime.fromtimestamp(collection_dir.stat().st_ctime),
                document_count=document_count,
                indexed=indexed,
            )

    async def upload_document(
        self, collection_id: str, file: UploadFile
    ) -> DocumentResponse:
        """Upload a document to a collection.

        Args:
            collection_id: The collection identifier
            file: The uploaded file

        Returns:
            DocumentResponse with document details

        Raises:
            ValueError: If collection does not exist
        """
        if self._is_cosmos_mode:
            # Check if collection exists
            if not self._collection_repo.get(collection_id):
                raise ValueError(f"Collection '{collection_id}' not found")

            # Read file content
            content = await file.read()
            filename = file.filename or "unnamed"

            # Store in Cosmos
            record = self._document_repo.put(collection_id, filename, content)

            return DocumentResponse(
                name=record.name,
                size=record.size,
                uploaded_at=record.uploaded_at,
            )
        else:
            # File mode
            collection_dir = self.collections_dir / collection_id

            if not collection_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")

            input_dir = collection_dir / "input"
            filename = file.filename or "unnamed"
            file_path = input_dir / filename

            # Save the file
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            return DocumentResponse(
                name=filename,
                size=file_path.stat().st_size,
                uploaded_at=datetime.now(),
            )

    def list_documents(self, collection_id: str) -> List[DocumentResponse]:
        """List all documents in a collection.

        Args:
            collection_id: The collection identifier

        Returns:
            List of DocumentResponse objects

        Raises:
            ValueError: If collection does not exist
        """
        if self._is_cosmos_mode:
            # Check if collection exists
            if not self._collection_repo.get(collection_id):
                raise ValueError(f"Collection '{collection_id}' not found")

            records = self._document_repo.list(collection_id)
            return [
                DocumentResponse(
                    name=rec.name,
                    size=rec.size,
                    uploaded_at=rec.uploaded_at,
                )
                for rec in records
            ]
        else:
            collection_dir = self.collections_dir / collection_id

            if not collection_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")

            input_dir = collection_dir / "input"
            documents = []

            if input_dir.exists():
                for file_path in input_dir.iterdir():
                    if file_path.is_file():
                        documents.append(
                            DocumentResponse(
                                name=file_path.name,
                                size=file_path.stat().st_size,
                                uploaded_at=datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ),
                            )
                        )

            return documents

    def delete_document(self, collection_id: str, document_name: str) -> bool:
        """Delete a document from a collection.

        Args:
            collection_id: The collection identifier
            document_name: The document filename

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If collection or document does not exist
        """
        if self._is_cosmos_mode:
            # Check if collection exists
            if not self._collection_repo.get(collection_id):
                raise ValueError(f"Collection '{collection_id}' not found")

            self._document_repo.delete(collection_id, document_name)
            return True
        else:
            collection_dir = self.collections_dir / collection_id

            if not collection_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")

            file_path = collection_dir / "input" / document_name

            if not file_path.exists():
                raise ValueError(f"Document '{document_name}' not found")

            file_path.unlink()
            return True

    def get_collection_path(self, collection_id: str) -> Dict[str, Path]:
        """Get paths for collection directories.

        Args:
            collection_id: The collection identifier

        Returns:
            Dictionary with paths for input, output, cache directories
        """
        collection_dir = self.collections_dir / collection_id

        return {
            "root": collection_dir,
            "input": collection_dir / "input",
            "output": collection_dir / "output",
            "cache": collection_dir / "cache",
        }


# Global storage service instance (lazy initialization)
_storage_service_instance: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get or create the global storage service instance."""
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = StorageService()
    return _storage_service_instance


# Backward compatibility - lazy property
class _LazyStorageService:
    """Lazy wrapper for storage service to delay initialization."""

    def __getattr__(self, name: str):
        return getattr(get_storage_service(), name)

    def __setattr__(self, name: str, value):
        return setattr(get_storage_service(), name, value)


storage_service = _LazyStorageService()
