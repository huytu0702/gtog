from dataclasses import dataclass
from datetime import datetime


@dataclass
class CollectionRecord:
    id: str
    name: str
    description: str | None
    created_at: datetime


@dataclass
class DocumentRecord:
    collection_id: str
    name: str
    size: int
    uploaded_at: datetime


@dataclass
class PromptRecord:
    collection_id: str
    prompt_name: str
    content: str
