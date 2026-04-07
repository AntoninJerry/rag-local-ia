from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_indexing_service, get_query_service
from app.api.main import app
from app.models import IndexingResult, QueryResponse, RetrievedChunk


class FakeIndexingService:
    def index_directory(self, source_dir: Path) -> IndexingResult:
        return IndexingResult(
            source_path=source_dir,
            files_read=1,
            files_ignored=0,
            indexed_documents=1,
            indexed_chunks=2,
            duration_seconds=0.25,
        )


class FakeQueryService:
    def ask(self, request) -> QueryResponse:
        return QueryResponse(
            answer=f"Reponse API pour: {request.question}",
            sources=[
                RetrievedChunk(
                    chunk_id="guide.md:0:0",
                    source_file="guide.md",
                    file_path=Path("data/documents/guide.md"),
                    text="Extrait utilise par l'API.",
                    chunk_index=0,
                    score=0.88,
                )
            ],
        )


def test_index_endpoint_delegates_to_indexing_service() -> None:
    app.dependency_overrides[get_indexing_service] = lambda: FakeIndexingService()
    client = TestClient(app)

    response = client.post("/index", json={"source_path": "data/documents"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_path"] == "data/documents"
    assert payload["files_read"] == 1
    assert payload["indexed_chunks"] == 2


def test_ask_endpoint_delegates_to_query_service() -> None:
    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Question API", "top_k": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Reponse API pour: Question API"
    assert payload["sources"][0]["chunk_id"] == "guide.md:0:0"
    assert payload["sources"][0]["score"] == 0.88


def test_ask_endpoint_validates_request_body() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": "", "top_k": 1})

    assert response.status_code == 422
