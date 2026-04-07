from pathlib import Path

from pydantic import BaseModel, Field, field_serializer

from app.core.config import settings


class DocumentRaw(BaseModel):
    """Text extracted from one source document or one page when available."""

    source_file: str
    file_path: Path
    text: str
    page_number: int | None = None

    @field_serializer("file_path")
    def serialize_file_path(self, file_path: Path) -> str:
        return file_path.as_posix()


class DocumentChunk(BaseModel):
    """Chunk ready for embedding and vector storage."""

    chunk_id: str
    source_file: str
    file_path: Path
    text: str
    chunk_index: int
    page_number: int | None = None

    @field_serializer("file_path")
    def serialize_file_path(self, file_path: Path) -> str:
        return file_path.as_posix()


class RetrievedChunk(DocumentChunk):
    """Chunk returned by retrieval with its similarity score."""

    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.retrieval_top_k, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    errors: list[str] = Field(default_factory=list)


class IndexingResult(BaseModel):
    source_path: Path
    files_read: int = 0
    files_ignored: int = 0
    indexed_documents: int = 0
    indexed_chunks: int = 0
    duration_seconds: float = 0.0
    skipped_files: list[Path] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_serializer("source_path")
    def serialize_source_path(self, source_path: Path) -> str:
        return source_path.as_posix()

    @field_serializer("skipped_files")
    def serialize_skipped_files(self, skipped_files: list[Path]) -> list[str]:
        return [path.as_posix() for path in skipped_files]
