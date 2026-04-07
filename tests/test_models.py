from pathlib import Path

from app.models import DocumentChunk, DocumentRaw, IndexingResult, QueryRequest, QueryResponse, RetrievedChunk


def test_document_raw_contains_source_metadata() -> None:
    document = DocumentRaw(
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        page_number=None,
        text="Contenu du document.",
    )

    assert document.source_file == "guide.md"
    assert document.file_path.as_posix() == "data/documents/guide.md"
    assert document.text == "Contenu du document."


def test_document_chunk_contains_chunk_metadata() -> None:
    chunk = DocumentChunk(
        chunk_id="guide.md:0:0",
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        chunk_index=0,
        text="Chunk.",
    )

    assert chunk.chunk_id == "guide.md:0:0"
    assert chunk.page_number is None


def test_retrieved_chunk_contains_score() -> None:
    chunk = RetrievedChunk(
        chunk_id="guide.md:0:0",
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        chunk_index=0,
        text="Chunk.",
        score=0.92,
    )

    assert chunk.score == 0.92


def test_query_models_describe_request_and_response() -> None:
    request = QueryRequest(question="Quelle est la source ?", top_k=3)
    response = QueryResponse(answer="Reponse non connectee.", sources=[])

    assert request.top_k == 3
    assert response.sources == []


def test_indexing_result_has_safe_defaults() -> None:
    result = IndexingResult(source_path=Path("data/documents"))

    assert result.indexed_documents == 0
    assert result.indexed_chunks == 0
    assert result.skipped_files == []
    assert result.errors == []
