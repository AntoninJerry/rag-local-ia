import json
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.core.config import settings
from app.models import DocumentChunk, RetrievedChunk
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the local vector store cannot complete an operation."""


class StoredVector(BaseModel):
    chunk: DocumentChunk
    embedding: list[float]


class LocalJsonVectorStore(VectorStore):
    """Small persistent vector store for local mono-user MVP usage."""

    def __init__(self, storage_dir: Path = settings.vector_store_dir) -> None:
        self.storage_dir = storage_dir
        self.index_path = self.storage_dir / "index.json"
        self._entries: dict[str, StoredVector] = {}
        self.load()

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings must have the same length.")
        if not chunks:
            return

        expected_dimension = self._current_dimension()
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector = self._normalize_embedding(embedding)
            if expected_dimension is None:
                expected_dimension = len(vector)
            if len(vector) != expected_dimension:
                raise VectorStoreError("All embeddings in the store must have the same dimension.")

            self._entries[chunk.chunk_id] = StoredVector(chunk=chunk, embedding=vector)

        self.persist()
        logger.info("Persisted %s vector store entrie(s)", len(self._entries))

    def load(self) -> None:
        if not self.index_path.exists():
            self._entries = {}
            return

        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            entries = [StoredVector.model_validate(item) for item in payload.get("entries", [])]
            self._entries = {entry.chunk.chunk_id: entry for entry in entries}
            logger.info("Loaded %s vector store entrie(s)", len(self._entries))
        except Exception as exc:
            raise VectorStoreError(f"Failed to load vector store index: {exc}") from exc

    def persist(self) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "entries": [entry.model_dump(mode="json") for entry in self._entries.values()],
            }
            self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            raise VectorStoreError(f"Failed to persist vector store index: {exc}") from exc

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")
        if not self._entries:
            return []

        query_vector = np.asarray(self._normalize_embedding(query_embedding), dtype=np.float32)
        expected_dimension = self._current_dimension()
        if expected_dimension is not None and query_vector.size != expected_dimension:
            raise VectorStoreError("Query embedding dimension does not match the vector store.")

        scored_chunks: list[RetrievedChunk] = []
        for entry in self._entries.values():
            score = self._cosine_similarity(query_vector, np.asarray(entry.embedding, dtype=np.float32))
            scored_chunks.append(RetrievedChunk(**entry.chunk.model_dump(), score=score))

        return sorted(scored_chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    def _current_dimension(self) -> int | None:
        first_entry = next(iter(self._entries.values()), None)
        if first_entry is None:
            return None
        return len(first_entry.embedding)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        if not embedding:
            raise VectorStoreError("Embedding must not be empty.")
        return [float(value) for value in embedding]

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator == 0:
            return 0.0
        return float(np.dot(left, right) / denominator)
