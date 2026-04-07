import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from app.chunking.chunker import TextChunker
from app.embeddings.base import EmbeddingProvider
from app.ingestion.service import DocumentIngestionService
from app.models import DocumentChunk, IndexingResult
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    """Orchestrates the full local indexing workflow for API and CLI callers."""

    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.timer = timer

    def index_directory(self, source_dir: Path) -> IndexingResult:
        start_time = self.timer()
        logger.info("Starting indexing pipeline for %s", source_dir)

        errors: list[str] = []
        skipped_files: list[Path] = []
        chunks: list[DocumentChunk] = []
        files_read = 0
        indexed_documents = 0

        try:
            ingestion_result = self.ingestion_service.ingest_directory(source_dir)
            errors.extend(ingestion_result.errors)
            skipped_files.extend(ingestion_result.ignored_files)
            skipped_files.extend(ingestion_result.failed_files)
            indexed_documents = len(ingestion_result.documents)
            files_read = len({document.file_path for document in ingestion_result.documents})

            for document in ingestion_result.documents:
                try:
                    document_chunks = self.chunker.split(document)
                    if not document_chunks:
                        skipped_files.append(document.file_path)
                        logger.warning("No chunk produced for document: %s", document.file_path)
                        continue
                    chunks.extend(document_chunks)
                except Exception as exc:
                    message = f"Chunking failed for {document.file_path}: {exc}"
                    errors.append(message)
                    skipped_files.append(document.file_path)
                    logger.exception("Chunking failed for document: %s", document.file_path)

            if chunks:
                try:
                    embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
                    self.vector_store.add_chunks(chunks, embeddings)
                except Exception as exc:
                    message = f"Vector indexing failed: {exc}"
                    errors.append(message)
                    logger.exception("Vector indexing failed for %s chunk(s)", len(chunks))

            logger.info(
                "Finished indexing pipeline: files_read=%s files_ignored=%s chunks=%s",
                files_read,
                len(set(skipped_files)),
                len(chunks),
            )
        except Exception as exc:
            message = f"Indexing failed: {exc}"
            errors.append(message)
            logger.exception("Indexing pipeline failed for %s", source_dir)

        duration_seconds = self.timer() - start_time
        return IndexingResult(
            source_path=source_dir,
            files_read=files_read,
            files_ignored=len(set(skipped_files)),
            indexed_documents=indexed_documents,
            indexed_chunks=len(chunks),
            duration_seconds=duration_seconds,
            skipped_files=sorted(set(skipped_files)),
            errors=errors,
        )
