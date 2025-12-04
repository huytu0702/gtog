"""File storage management service."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import UploadFile

from ..config import settings
from ..models import CollectionResponse, DocumentResponse


class StorageService:
    """Service for managing file storage operations."""
    
    def __init__(self):
        """Initialize the storage service."""
        self.collections_dir = settings.collections_dir
        # Ensure collections directory exists
        self.collections_dir.mkdir(parents=True, exist_ok=True)
    
    def create_collection(self, collection_id: str, description: Optional[str] = None) -> CollectionResponse:
        """
        Create a new collection with its directory structure.
        
        Args:
            collection_id: Unique identifier for the collection
            description: Optional description of the collection
            
        Returns:
            CollectionResponse with collection details
            
        Raises:
            ValueError: If collection already exists
        """
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
            # Import index prompts
            from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
            from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
            from graphrag.prompts.index.community_report_text_units import COMMUNITY_REPORT_TEXT_PROMPT
            from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
            from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
            
            # Import query prompts
            from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
            from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT
            from graphrag.prompts.query.global_search_knowledge_system_prompt import GENERAL_KNOWLEDGE_INSTRUCTION
            from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
            from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
            from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
            from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT
            
            prompts_dir = collection_dir / "prompts"
            
            # Write index prompts
            (prompts_dir / "extract_graph.txt").write_text(GRAPH_EXTRACTION_PROMPT, encoding="utf-8")
            (prompts_dir / "summarize_descriptions.txt").write_text(SUMMARIZE_PROMPT, encoding="utf-8")
            (prompts_dir / "extract_claims.txt").write_text(EXTRACT_CLAIMS_PROMPT, encoding="utf-8")
            (prompts_dir / "community_report_graph.txt").write_text(COMMUNITY_REPORT_PROMPT, encoding="utf-8")
            (prompts_dir / "community_report_text.txt").write_text(COMMUNITY_REPORT_TEXT_PROMPT, encoding="utf-8")
            
            # Write query prompts
            (prompts_dir / "drift_search_system_prompt.txt").write_text(DRIFT_LOCAL_SYSTEM_PROMPT, encoding="utf-8")
            (prompts_dir / "drift_search_reduce_prompt.txt").write_text(DRIFT_REDUCE_PROMPT, encoding="utf-8")
            (prompts_dir / "global_search_map_system_prompt.txt").write_text(MAP_SYSTEM_PROMPT, encoding="utf-8")
            (prompts_dir / "global_search_reduce_system_prompt.txt").write_text(REDUCE_SYSTEM_PROMPT, encoding="utf-8")
            (prompts_dir / "global_search_knowledge_system_prompt.txt").write_text(GENERAL_KNOWLEDGE_INSTRUCTION, encoding="utf-8")
            (prompts_dir / "local_search_system_prompt.txt").write_text(LOCAL_SEARCH_SYSTEM_PROMPT, encoding="utf-8")
            (prompts_dir / "basic_search_system_prompt.txt").write_text(BASIC_SEARCH_SYSTEM_PROMPT, encoding="utf-8")
            (prompts_dir / "question_gen_system_prompt.txt").write_text(QUESTION_SYSTEM_PROMPT, encoding="utf-8")
        except Exception as e:
            import logging
            logging.warning(f"Failed to generate prompts for collection {collection_id}: {e}")
        
        return CollectionResponse(
            id=collection_id,
            name=collection_id,
            description=description,
            created_at=datetime.now(),
            document_count=0,
            indexed=False,
        )
    
    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection and all its contents.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            True if deleted successfully
            
        Raises:
            ValueError: If collection does not exist
        """
        collection_dir = self.collections_dir / collection_id
        
        if not collection_dir.exists():
            raise ValueError(f"Collection '{collection_id}' not found")
        
        shutil.rmtree(collection_dir)
        return True
    
    def list_collections(self) -> List[CollectionResponse]:
        """
        List all available collections.
        
        Returns:
            List of CollectionResponse objects
        """
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
                    document_count = len([f for f in input_dir.iterdir() if f.is_file()])
                
                # Check if indexed (has output files)
                indexed = False
                if output_dir.exists():
                    required_files = ["create_final_entities.parquet", "create_final_communities.parquet"]
                    indexed = all((output_dir / f).exists() for f in required_files)
                
                collections.append(CollectionResponse(
                    id=collection_id,
                    name=collection_id,
                    description=None,
                    created_at=datetime.fromtimestamp(collection_dir.stat().st_ctime),
                    document_count=document_count,
                    indexed=indexed,
                ))
        
        return collections
    
    def get_collection(self, collection_id: str) -> Optional[CollectionResponse]:
        """
        Get details about a specific collection.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            CollectionResponse or None if not found
        """
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
            required_files = ["create_final_entities.parquet", "create_final_communities.parquet"]
            indexed = all((output_dir / f).exists() for f in required_files)
        
        return CollectionResponse(
            id=collection_id,
            name=collection_id,
            description=None,
            created_at=datetime.fromtimestamp(collection_dir.stat().st_ctime),
            document_count=document_count,
            indexed=indexed,
        )
    
    async def upload_document(self, collection_id: str, file: UploadFile) -> DocumentResponse:
        """
        Upload a document to a collection.
        
        Args:
            collection_id: The collection identifier
            file: The uploaded file
            
        Returns:
            DocumentResponse with document details
            
        Raises:
            ValueError: If collection does not exist
        """
        collection_dir = self.collections_dir / collection_id
        
        if not collection_dir.exists():
            raise ValueError(f"Collection '{collection_id}' not found")
        
        input_dir = collection_dir / "input"
        file_path = input_dir / file.filename
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return DocumentResponse(
            name=file.filename,
            size=file_path.stat().st_size,
            uploaded_at=datetime.now(),
        )
    
    def list_documents(self, collection_id: str) -> List[DocumentResponse]:
        """
        List all documents in a collection.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            List of DocumentResponse objects
            
        Raises:
            ValueError: If collection does not exist
        """
        collection_dir = self.collections_dir / collection_id
        
        if not collection_dir.exists():
            raise ValueError(f"Collection '{collection_id}' not found")
        
        input_dir = collection_dir / "input"
        documents = []
        
        if input_dir.exists():
            for file_path in input_dir.iterdir():
                if file_path.is_file():
                    documents.append(DocumentResponse(
                        name=file_path.name,
                        size=file_path.stat().st_size,
                        uploaded_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                    ))
        
        return documents
    
    def delete_document(self, collection_id: str, document_name: str) -> bool:
        """
        Delete a document from a collection.
        
        Args:
            collection_id: The collection identifier
            document_name: The document filename
            
        Returns:
            True if deleted successfully
            
        Raises:
            ValueError: If collection or document does not exist
        """
        collection_dir = self.collections_dir / collection_id
        
        if not collection_dir.exists():
            raise ValueError(f"Collection '{collection_id}' not found")
        
        file_path = collection_dir / "input" / document_name
        
        if not file_path.exists():
            raise ValueError(f"Document '{document_name}' not found")
        
        file_path.unlink()
        return True
    
    def get_collection_path(self, collection_id: str) -> Dict[str, Path]:
        """
        Get paths for collection directories.
        
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


# Global storage service instance
storage_service = StorageService()
