from pathlib import Path

import pytest

from app.models import RetrievedChunk
from app.retrieval.retriever import RetrievalError, Retriever


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if "python" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_query_embedding: list[float] | None = None
        self.last_top_k: int | None = None

    def add_chunks(self, chunks, embeddings) -> None:
        return None

    def load(self) -> None:
        return None

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k
        chunks = [
            RetrievedChunk(
                chunk_id="python.md:0:0",
                source_file="python.md",
                file_path=Path("data/documents/python.md"),
                text="Python est pertinent.",
                chunk_index=0,
                score=query_embedding[0],
            ),
            RetrievedChunk(
                chunk_id="other.md:0:0",
                source_file="other.md",
                file_path=Path("data/documents/other.md"),
                text="Autre contenu.",
                chunk_index=0,
                score=query_embedding[1],
            ),
        ]
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]


def test_retriever_embeds_query_and_returns_most_relevant_chunks() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_provider=embedding_provider, vector_store=vector_store, default_top_k=1)

    results = retriever.retrieve("  question sur Python  ")

    assert embedding_provider.queries == ["question sur Python"]
    assert vector_store.last_query_embedding == [1.0, 0.0]
    assert vector_store.last_top_k == 1
    assert len(results) == 1
    assert results[0].chunk_id == "python.md:0:0"
    assert results[0].score == 1.0


def test_retriever_uses_explicit_top_k() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        default_top_k=1,
    )

    results = retriever.retrieve("autre question", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "other.md:0:0"


def test_retriever_rejects_empty_question_and_invalid_top_k() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")

    with pytest.raises(RetrievalError):
        retriever.retrieve("question", top_k=0)
