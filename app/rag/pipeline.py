import logging
from typing import Protocol

from app.core.config import settings
from app.llm.base import LLMClient
from app.llm.prompt_builder import RagPromptBuilder
from app.models import QueryResponse, RetrievedChunk

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "Je ne sais pas. Les documents indexes ne contiennent pas d'information suffisante "
    "pour repondre a cette question."
)
LLM_FAILURE_ANSWER = (
    "Je ne peux pas generer une reponse pour le moment, car le fournisseur LLM "
    "n'a pas repondu correctement. Les sources recuperees restent disponibles."
)


class RetrieverLike(Protocol):
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return retrieved chunks for a user question."""


class RagPipelineError(RuntimeError):
    """Raised when the RAG pipeline cannot complete a query."""


class RagPipeline:
    """Orchestrates retrieval, prompt building and LLM answer generation."""

    def __init__(
        self,
        retriever: RetrieverLike,
        llm_client: LLMClient,
        prompt_builder: RagPromptBuilder | None = None,
        default_top_k: int = settings.retrieval_top_k,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or RagPromptBuilder()
        self.default_top_k = default_top_k

    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise RagPipelineError("Question must not be empty.")

        effective_top_k = self.default_top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise RagPipelineError("top_k must be greater than 0.")

        logger.info("Starting RAG pipeline with top_k=%s", effective_top_k)
        retrieved_chunks = self.retriever.retrieve(cleaned_question, top_k=effective_top_k)
        if not retrieved_chunks:
            logger.info("No relevant context found for query")
            return QueryResponse(answer=NO_CONTEXT_ANSWER, sources=[])

        prompt = self.prompt_builder.build(cleaned_question, retrieved_chunks)
        try:
            answer = self.llm_client.generate_answer(prompt).strip()
        except Exception as exc:
            logger.exception("LLM provider failed during answer generation")
            return QueryResponse(answer=LLM_FAILURE_ANSWER, sources=retrieved_chunks, errors=[str(exc)])

        if not answer:
            logger.error("LLM provider returned an empty answer")
            return QueryResponse(
                answer=LLM_FAILURE_ANSWER,
                sources=retrieved_chunks,
                errors=["LLM provider returned an empty answer."],
            )

        logger.info("RAG pipeline completed with %s source chunk(s)", len(retrieved_chunks))
        return QueryResponse(answer=answer, sources=retrieved_chunks)
