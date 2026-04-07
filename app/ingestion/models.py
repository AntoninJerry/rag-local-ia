from pathlib import Path

from pydantic import BaseModel


class Document(BaseModel):
    text: str
    source_path: Path


class DocumentChunk(BaseModel):
    text: str
    source_path: Path
    chunk_index: int
