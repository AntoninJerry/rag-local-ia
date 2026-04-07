from pathlib import Path

import pytest

from app.llm.prompt_builder import RagPromptBuilder
from app.models import RetrievedChunk
from app.rag.pipeline import LLM_FAILURE_ANSWER, NO_CONTEXT_ANSWER, RagPipeline, RagPipelineError


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((question, top_k))
        return self.chunks[: top_k or len(self.chunks)]


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Reponse basee sur les sources."


class FailingLLMClient:
    def generate_answer(self, prompt: str) -> str:
        raise RuntimeError("LLM timeout")


class EmptyLLMClient:
    def generate_answer(self, prompt: str) -> str:
        return "   "


def make_chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        text="Le projet utilise un pipeline RAG local.",
        chunk_index=0,
        page_number=1,
        score=score,
    )


def test_rag_pipeline_retrieves_builds_prompt_and_returns_sources() -> None:
    retriever = FakeRetriever([make_chunk("guide.md:1:0", 0.91), make_chunk("guide.md:1:1", 0.82)])
    llm_client = FakeLLMClient()
    pipeline = RagPipeline(
        retriever=retriever,
        llm_client=llm_client,
        prompt_builder=RagPromptBuilder(),
        default_top_k=1,
    )

    response = pipeline.answer_question("  Comment fonctionne le projet ?  ")

    assert retriever.calls == [("Comment fonctionne le projet ?", 1)]
    assert response.answer == "Reponse basee sur les sources."
    assert [source.chunk_id for source in response.sources] == ["guide.md:1:0"]
    assert len(llm_client.prompts) == 1
    assert "Comment fonctionne le projet ?" in llm_client.prompts[0]
    assert "source_id: guide.md:1:0" in llm_client.prompts[0]


def test_rag_pipeline_returns_safe_answer_without_context() -> None:
    retriever = FakeRetriever([])
    llm_client = FakeLLMClient()
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    response = pipeline.answer_question("Question sans source")

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == []
    assert llm_client.prompts == []


def test_rag_pipeline_returns_safe_answer_when_llm_fails() -> None:
    pipeline = RagPipeline(
        retriever=FakeRetriever([make_chunk("guide.md:1:0", 0.91)]),
        llm_client=FailingLLMClient(),
    )

    response = pipeline.answer_question("Question avec source")

    assert response.answer == LLM_FAILURE_ANSWER
    assert response.sources[0].chunk_id == "guide.md:1:0"
    assert response.errors == ["LLM timeout"]


def test_rag_pipeline_returns_safe_answer_when_llm_returns_empty_text() -> None:
    pipeline = RagPipeline(
        retriever=FakeRetriever([make_chunk("guide.md:1:0", 0.91)]),
        llm_client=EmptyLLMClient(),
    )

    response = pipeline.answer_question("Question avec source")

    assert response.answer == LLM_FAILURE_ANSWER
    assert response.sources[0].chunk_id == "guide.md:1:0"
    assert response.errors == ["LLM provider returned an empty answer."]


def test_rag_pipeline_rejects_empty_question_and_invalid_top_k() -> None:
    pipeline = RagPipeline(retriever=FakeRetriever([]), llm_client=FakeLLMClient())

    with pytest.raises(RagPipelineError):
        pipeline.answer_question("   ")

    with pytest.raises(RagPipelineError):
        pipeline.answer_question("Question", top_k=0)
