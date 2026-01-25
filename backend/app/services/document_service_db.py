"""Database-backed document service."""

from datetime import datetime
from typing import List
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Document
from app.models import DocumentResponse
from app.repositories import CollectionRepository, DocumentRepository


class DocumentServiceDB:
    """Service for document operations using database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.collections = CollectionRepository(session)
        self.documents = DocumentRepository(session)

    async def _get_collection_or_error(self, collection_id: UUID) -> Collection:
        collection = await self.collections.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection '{collection_id}' not found")
        return collection

    async def upload_document(self, collection_id: UUID, file: UploadFile) -> DocumentResponse:
        await self._get_collection_or_error(collection_id)

        # Read file bytes
        content = await file.read()

        # Create document row
        doc = Document(
            collection_id=collection_id,
            title=file.filename,
            text=None,
            doc_metadata=None,
            filename=file.filename,
            content_type=file.content_type,
            bytes_content=content,
        )
        await self.documents.create(doc)

        return DocumentResponse(
            name=doc.filename,
            size=len(content),
            uploaded_at=datetime.now(),
        )

    async def list_documents(self, collection_id: UUID) -> List[DocumentResponse]:
        await self._get_collection_or_error(collection_id)
        docs = await self.documents.get_by_collection(collection_id)
        return [
            DocumentResponse(
                name=doc.filename,
                size=len(doc.bytes_content or b""),
                uploaded_at=doc.created_at,
            )
            for doc in docs
        ]

    async def delete_document(self, collection_id: UUID, document_name: str) -> None:
        await self._get_collection_or_error(collection_id)
        deleted = await self.documents.delete_by_name(collection_id, document_name)
        if not deleted:
            raise ValueError(f"Document '{document_name}' not found")
