from typing import Protocol

from app.ingestion.models import DocumentChunk


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings locally."""

    def search(self, query_embedding: list[float], top_k: int) -> list[DocumentChunk]:
        """Return the most similar chunks for a query embedding."""
