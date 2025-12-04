"""Document management endpoints."""

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..models import DocumentResponse, DocumentList
from ..services import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(collection_id: str, file: UploadFile = File(...)):
    """Upload a document to a collection."""
    try:
        result = await storage_service.upload_document(collection_id, file)
        logger.info(f"Uploaded document {file.filename} to collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error uploading document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=DocumentList)
async def list_documents(collection_id: str):
    """List all documents in a collection."""
    try:
        documents = storage_service.list_documents(collection_id)
        return DocumentList(documents=documents, total=len(documents))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{document_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(collection_id: str, document_name: str):
    """Delete a document from a collection."""
    try:
        storage_service.delete_document(collection_id, document_name)
        logger.info(f"Deleted document {document_name} from collection {collection_id}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
