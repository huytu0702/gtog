"""Document management endpoints."""

import logging
from uuid import UUID
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ..api.deps import get_document_service
from ..models import DocumentResponse, DocumentList
from ..services.document_service_db import DocumentServiceDB

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


router = APIRouter(prefix="/api/collections/{collection_id}/documents", tags=["documents"])


def _enforce_upload_size(file: UploadFile) -> None:
    """Validate file size against MAX_UPLOAD_BYTES."""
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds maximum size of 25 MB",
        )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    service: DocumentServiceDB = Depends(get_document_service),
):
    """Upload a document to a collection."""
    try:
        _enforce_upload_size(file)

        # Validate file extension
        if not file.filename.lower().endswith(('.txt', '.md')):
            raise ValueError("Only .txt and .md files are supported")

        result = await service.upload_document(UUID(collection_id), file)
        logger.info(f"Uploaded document {file.filename} to collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=DocumentList)
async def list_documents(
    collection_id: str,
    service: DocumentServiceDB = Depends(get_document_service),
):
    """List all documents in a collection."""
    try:
        documents = await service.list_documents(UUID(collection_id))
        return DocumentList(documents=documents, total=len(documents))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{document_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    collection_id: str,
    document_name: str,
    service: DocumentServiceDB = Depends(get_document_service),
):
    """Delete a document from a collection."""
    try:
        await service.delete_document(UUID(collection_id), document_name)
        logger.info(f"Deleted document {document_name} from collection {collection_id}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
