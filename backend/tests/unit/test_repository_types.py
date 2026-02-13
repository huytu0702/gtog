import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.repositories.types import CollectionRecord, DocumentRecord, PromptRecord


def test_collection_record_fields():
    rec = CollectionRecord(id="demo", name="demo", description=None, created_at=datetime.utcnow())
    assert rec.id == "demo"


def test_document_record_fields():
    rec = DocumentRecord(collection_id="demo", name="a.txt", size=12, uploaded_at=datetime.utcnow())
    assert rec.name.endswith(".txt")


def test_prompt_record_fields():
    rec = PromptRecord(collection_id="demo", prompt_name="local_search_system_prompt.txt", content="text")
    assert rec.prompt_name.endswith(".txt")
