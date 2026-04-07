from pathlib import Path

import pytest

from app.models import QueryRequest, QueryResponse, RetrievedChunk
from app.querying.service import QueryService, QueryServiceError


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        self.calls.append((question, top_k))
        return QueryResponse(
            answer="Reponse structuree.",
            sources=[
                RetrievedChunk(
                    chunk_id="guide.md:0:0",
                    source_file="guide.md",
                    file_path=Path("data/documents/guide.md"),
                    text="Extrait pertinent.",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


def test_query_service_calls_pipeline_and_returns_structured_response() -> None:
    pipeline = FakePipeline()
    service = QueryService(pipeline=pipeline)

    response = service.ask(QueryRequest(question="  Quelle est la source ?  ", top_k=3))

    assert pipeline.calls == [("Quelle est la source ?", 3)]
    assert response.answer == "Reponse structuree."
    assert response.sources[0].chunk_id == "guide.md:0:0"
    assert response.sources[0].score == 0.9


def test_query_service_rejects_empty_question() -> None:
    service = QueryService(pipeline=FakePipeline())

    with pytest.raises(QueryServiceError):
        service.ask(QueryRequest.model_construct(question="   ", top_k=3))
