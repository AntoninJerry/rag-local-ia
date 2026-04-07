from pathlib import Path

from app.chunking.chunker import TextChunker
from app.indexing.service import IndexingService
from app.ingestion.models import IngestionResult
from app.models import DocumentRaw


class FakeIngestionService:
    def ingest_directory(self, source_dir: Path) -> IngestionResult:
        return IngestionResult(
            documents=[
                DocumentRaw(
                    source_file="guide.md",
                    file_path=source_dir / "guide.md",
                    text="Premier document. Deuxieme phrase.",
                ),
                DocumentRaw(
                    source_file="empty.md",
                    file_path=source_dir / "empty.md",
                    text="   ",
                ),
            ],
            ignored_files=[source_dir / "ignored.docx"],
            failed_files=[source_dir / "broken.pdf"],
            errors=[f"{source_dir / 'broken.pdf'}: unreadable"],
        )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_chunks = []
        self.added_embeddings = []

    def add_chunks(self, chunks, embeddings) -> None:
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def load(self) -> None:
        return None

    def search(self, query_embedding, top_k):
        return []


class FakeTimer:
    def __init__(self) -> None:
        self.values = iter([10.0, 12.5])

    def __call__(self) -> float:
        return next(self.values)


class FailingChunker:
    def split(self, document: DocumentRaw):
        if document.source_file == "broken_chunk.md":
            raise RuntimeError("chunker exploded")
        return TextChunker(chunk_size=40, overlap=5).split(document)


class ChunkFailureIngestionService:
    def ingest_directory(self, source_dir: Path) -> IngestionResult:
        return IngestionResult(
            documents=[
                DocumentRaw(
                    source_file="ok.md",
                    file_path=source_dir / "ok.md",
                    text="Document valide.",
                ),
                DocumentRaw(
                    source_file="broken_chunk.md",
                    file_path=source_dir / "broken_chunk.md",
                    text="Document qui casse.",
                ),
            ]
        )


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


def test_indexing_service_orchestrates_ingestion_chunking_embeddings_and_store(tmp_path: Path) -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    service = IndexingService(
        ingestion_service=FakeIngestionService(),
        chunker=TextChunker(chunk_size=40, overlap=5),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        timer=FakeTimer(),
    )

    result = service.index_directory(tmp_path)

    assert result.files_read == 2
    assert result.files_ignored == 3
    assert result.indexed_documents == 2
    assert result.indexed_chunks == 1
    assert result.duration_seconds == 2.5
    assert tmp_path / "ignored.docx" in result.skipped_files
    assert tmp_path / "broken.pdf" in result.skipped_files
    assert tmp_path / "empty.md" in result.skipped_files
    assert len(result.errors) == 1
    assert embedding_provider.texts == ["Premier document. Deuxieme phrase."]
    assert len(vector_store.added_chunks) == 1
    assert vector_store.added_embeddings == [[0.0, 1.0]]


def test_indexing_service_continues_when_one_document_chunking_fails(tmp_path: Path) -> None:
    vector_store = FakeVectorStore()
    service = IndexingService(
        ingestion_service=ChunkFailureIngestionService(),
        chunker=FailingChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = service.index_directory(tmp_path)

    assert result.indexed_documents == 2
    assert result.indexed_chunks == 1
    assert result.files_ignored == 1
    assert tmp_path / "broken_chunk.md" in result.skipped_files
    assert "Chunking failed" in result.errors[0]
    assert len(vector_store.added_chunks) == 1


def test_indexing_service_reports_embedding_failure_without_crashing(tmp_path: Path) -> None:
    vector_store = FakeVectorStore()
    service = IndexingService(
        ingestion_service=ChunkFailureIngestionService(),
        chunker=TextChunker(chunk_size=40, overlap=5),
        embedding_provider=FailingEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = service.index_directory(tmp_path)

    assert result.indexed_chunks == 2
    assert result.errors == ["Vector indexing failed: embedding backend unavailable"]
    assert vector_store.added_chunks == []
