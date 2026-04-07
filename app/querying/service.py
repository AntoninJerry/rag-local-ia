import logging
from typing import Protocol

from app.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class QueryPipeline(Protocol):
    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        """Answer a user question with a structured RAG response."""


class QueryServiceError(ValueError):
    """Raised when a query request cannot be handled."""


class QueryService:
    """UI-agnostic question answering service for CLI and API callers."""

    def __init__(self, pipeline: QueryPipeline) -> None:
        self.pipeline = pipeline

    def ask(self, request: QueryRequest) -> QueryResponse:
        question = request.question.strip()
        if not question:
            raise QueryServiceError("Question must not be empty.")

        logger.info("Handling query with top_k=%s", request.top_k)
        return self.pipeline.answer_question(question=question, top_k=request.top_k)
