"""Indexing service for GraphRAG operations."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

from ..config import settings
from ..models import IndexStatus, IndexStatusResponse
from ..utils import load_graphrag_config

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for managing indexing operations."""
    
    def __init__(self):
        """Initialize the indexing service."""
        self.indexing_tasks: Dict[str, IndexStatusResponse] = {}
    
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
            logger.info(f"Starting indexing for collection: {collection_id}")
            
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
