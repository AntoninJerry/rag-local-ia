from collections.abc import Callable
from typing import Any

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider

ModelFactory = Callable[[str], Any]


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce embeddings."""


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider backed by sentence-transformers.

    To replace this backend later, for example with Ollama embeddings, create a
    new class that implements EmbeddingProvider and keep the same public methods:
    embed_documents(texts) and embed_query(text).
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text.strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise EmbeddingProviderError("Document texts must not be empty.")
        if not cleaned_texts:
            return []

        try:
            embeddings = self._get_model().encode(cleaned_texts, convert_to_numpy=False)
            return self._to_float_vectors(embeddings)
        except Exception as exc:
            raise EmbeddingProviderError(f"Failed to embed documents: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise EmbeddingProviderError("Query text must not be empty.")

        try:
            embedding = self._get_model().encode(cleaned_text, convert_to_numpy=False)
            return self._to_float_vector(embedding)
        except Exception as exc:
            raise EmbeddingProviderError(f"Failed to embed query: {exc}") from exc

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory(self.model_name)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderError(
                "sentence-transformers is not installed. "
                'Install it with: python -m pip install -e ".[embeddings-local]"'
            ) from exc

        return SentenceTransformer(self.model_name)

    @classmethod
    def _to_float_vectors(cls, embeddings: Any) -> list[list[float]]:
        return [cls._to_float_vector(embedding) for embedding in embeddings]

    @staticmethod
    def _to_float_vector(embedding: Any) -> list[float]:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        return [float(value) for value in embedding]
