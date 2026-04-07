from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import settings


class DocumentRaw(BaseModel):
    """Text extracted from one source document or one page when available."""

    source_file: str
    file_path: Path
    text: str
    page_number: int | None = None


class DocumentChunk(BaseModel):
    """Chunk ready for embedding and vector storage."""

    chunk_id: str
    source_file: str
    file_path: Path
    text: str
    chunk_index: int
    page_number: int | None = None


class RetrievedChunk(DocumentChunk):
    """Chunk returned by retrieval with its similarity score."""

    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.retrieval_top_k, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]


class IndexingResult(BaseModel):
    source_path: Path
    indexed_documents: int = 0
    indexed_chunks: int = 0
    skipped_files: list[Path] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
