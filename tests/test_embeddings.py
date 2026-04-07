import pytest

from app.embeddings.sentence_transformers import (
    EmbeddingProviderError,
    SentenceTransformersEmbeddingProvider,
)


class FakeSentenceTransformer:
    load_count = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        FakeSentenceTransformer.load_count += 1

    def encode(self, texts, convert_to_numpy: bool = False):
        if isinstance(texts, str):
            return [1, len(texts)]
        return [[index + 1, len(text)] for index, text in enumerate(texts)]


def test_embed_documents_returns_vectors_and_loads_model_once() -> None:
    FakeSentenceTransformer.load_count = 0
    provider = SentenceTransformersEmbeddingProvider(
        model_name="fake-model",
        model_factory=FakeSentenceTransformer,
    )

    first = provider.embed_documents(["alpha", "beta"])
    second = provider.embed_documents(["gamma"])

    assert first == [[1.0, 5.0], [2.0, 4.0]]
    assert second == [[1.0, 5.0]]
    assert FakeSentenceTransformer.load_count == 1


def test_embed_query_returns_single_vector() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        model_name="fake-model",
        model_factory=FakeSentenceTransformer,
    )

    assert provider.embed_query("question") == [1.0, 8.0]


def test_embedding_provider_rejects_empty_text() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        model_name="fake-model",
        model_factory=FakeSentenceTransformer,
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("   ")

    with pytest.raises(EmbeddingProviderError):
        provider.embed_documents(["valid", "   "])
