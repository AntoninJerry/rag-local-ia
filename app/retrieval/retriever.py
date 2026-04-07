import logging

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider
from app.models import RetrievedChunk
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class RetrievalError(ValueError):
    """Raised when retrieval cannot be executed for an invalid request."""


class Retriever:
    """Coordinates vector search for a user question."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = settings.retrieval_top_k,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise RetrievalError("Question must not be empty.")

        effective_top_k = self.default_top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")

        logger.info("Retrieving top %s chunk(s) for query", effective_top_k)
        query_embedding = self.embedding_provider.embed_query(cleaned_question)
        return self.vector_store.search(query_embedding=query_embedding, top_k=effective_top_k)
