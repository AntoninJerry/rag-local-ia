from typing import Protocol

from app.models import DocumentChunk, RetrievedChunk


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings locally."""

    def load(self) -> None:
        """Reload the persisted index from local storage."""

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the most similar chunks for a query embedding."""
