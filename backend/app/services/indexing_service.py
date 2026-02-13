"""Indexing service for GraphRAG operations."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

from ..config import settings
from ..models import IndexStatus, IndexStatusResponse
from ..utils import load_graphrag_config
from ..utils.arrow_fix import apply_arrow_fix, remove_arrow_fix

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for managing indexing operations."""
    
    def __init__(self):
        """Initialize the indexing service."""
        self.indexing_tasks: Dict[str, IndexStatusResponse] = {}
        self._document_repo: Optional[object] = None
    
    def _get_document_repo(self) -> Optional[object]:
        """Get the document repository from storage service if in cosmos mode."""
        if not settings.is_cosmos_mode:
            return None
        
        if self._document_repo is None:
            # Import here to avoid circular dependency
            from .storage_service import get_storage_service
            storage_svc = get_storage_service()
            if hasattr(storage_svc, '_document_repo'):
                self._document_repo = storage_svc._document_repo
        
        return self._document_repo
    
    def _sync_cosmos_documents_to_input(self, collection_id: str, input_dir: Path) -> None:
        """
        Sync documents from Cosmos DB to local input directory for GraphRAG indexing.
        
        This is needed because GraphRAG indexing reads from local filesystem.
        Operation is idempotent - overwrites existing files each run.
        
        Args:
            collection_id: The collection identifier
            input_dir: Path to the input directory
        """
        doc_repo = self._get_document_repo()
        if doc_repo is None:
            logger.debug("Not in cosmos mode, skipping document sync")
            return
        
        logger.info(f"Syncing cosmos documents to input dir for collection: {collection_id}")
        
        # Ensure input directory exists
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all documents for this collection
        documents = doc_repo.list(collection_id)
        
        synced_count = 0
        for doc in documents:
            try:
                # Get document content from Cosmos
                content = doc_repo.get_content(collection_id, doc.name)
                if content is None:
                    logger.warning(f"Could not retrieve content for document: {doc.name}")
                    continue
                
                # Write to local input directory (overwrite if exists)
                file_path = input_dir / doc.name
                file_path.write_bytes(content)
                synced_count += 1
                logger.debug(f"Synced document: {doc.name}")
            except Exception as e:
                logger.error(f"Failed to sync document {doc.name}: {e}")
        
        logger.info(f"Synced {synced_count} documents to input directory")
    
    async def start_indexing(self, collection_id: str) -> IndexStatusResponse:
        """
        Start indexing a collection in the background.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            IndexStatusResponse with initial status
        """
        # Check if already indexing
        if collection_id in self.indexing_tasks:
            current_status = self.indexing_tasks[collection_id]
            if current_status.status == IndexStatus.RUNNING:
                return current_status
        
        # Initialize status
        status_response = IndexStatusResponse(
            collection_id=collection_id,
            status=IndexStatus.RUNNING,
            progress=0.0,
            message="Starting indexing...",
            started_at=datetime.now(),
        )
        self.indexing_tasks[collection_id] = status_response
        
        # Start indexing task in background
        asyncio.create_task(self._run_indexing_task(collection_id))
        
        return status_response
    
    async def _run_indexing_task(self, collection_id: str):
        """
        Internal task for running the indexing process.

        Args:
            collection_id: The collection identifier
        """
        try:
            # Apply ArrowStringArray fix before indexing
            apply_arrow_fix()

            logger.info(f"Starting indexing for collection: {collection_id}")
            
            # Update status
            self.indexing_tasks[collection_id].message = "Preparing input files..."
            self.indexing_tasks[collection_id].progress = 5.0
            
            # If in cosmos mode, sync documents to local input directory
            if settings.is_cosmos_mode:
                input_dir = settings.collections_dir / collection_id / "input"
                self._sync_cosmos_documents_to_input(collection_id, input_dir)
            
            # Update status
            self.indexing_tasks[collection_id].message = "Loading configuration..."
            self.indexing_tasks[collection_id].progress = 10.0
            
            # Load GraphRAG config with collection-specific overrides
            config = load_graphrag_config(collection_id)
            
            logger.info(f"Configuration loaded for {collection_id}")
            
            # Update status
            self.indexing_tasks[collection_id].message = "Running indexing pipeline..."
            self.indexing_tasks[collection_id].progress = 20.0
            
            # Run the indexing pipeline
            outputs = await api.build_index(
                config=config,
                verbose=True,
                callbacks=[NoopWorkflowCallbacks()],
            )
            
            # Check for errors
            has_errors = any(output.errors and len(output.errors) > 0 for output in outputs)
            
            if has_errors:
                error_messages = []
                for output in outputs:
                    if output.errors:
                        error_messages.extend([str(e) for e in output.errors])
                
                self.indexing_tasks[collection_id].status = IndexStatus.FAILED
                self.indexing_tasks[collection_id].error = "; ".join(error_messages[:3])  # Limit error messages
                self.indexing_tasks[collection_id].message = "Indexing failed"
                logger.error(f"Indexing failed for {collection_id}: {error_messages}")
            else:
                self.indexing_tasks[collection_id].status = IndexStatus.COMPLETED
                self.indexing_tasks[collection_id].progress = 100.0
                self.indexing_tasks[collection_id].message = "Indexing completed successfully"
                self.indexing_tasks[collection_id].completed_at = datetime.now()
                logger.info(f"Indexing completed successfully for {collection_id}")
        
        except Exception as e:
            logger.exception(f"Error during indexing for {collection_id}")
            self.indexing_tasks[collection_id].status = IndexStatus.FAILED
            self.indexing_tasks[collection_id].error = str(e)
            self.indexing_tasks[collection_id].message = "Indexing failed with error"
        finally:
            # Always remove the patch after indexing
            remove_arrow_fix()

    def get_index_status(self, collection_id: str) -> Optional[IndexStatusResponse]:
        """
        Get the current indexing status for a collection.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            IndexStatusResponse or None if never indexed
        """
        return self.indexing_tasks.get(collection_id)


# Global indexing service instance
indexing_service = IndexingService()
