from pathlib import Path

from pydantic import BaseModel, Field

from app.models import DocumentChunk, DocumentRaw

Document = DocumentRaw


class IngestionResult(BaseModel):
    documents: list[DocumentRaw] = Field(default_factory=list)
    ignored_files: list[Path] = Field(default_factory=list)
    failed_files: list[Path] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
