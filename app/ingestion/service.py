import logging
from pathlib import Path

from app.ingestion.loaders import DocumentLoader, default_loaders
from app.ingestion.models import IngestionResult
from app.ingestion.scanner import scan_documents
from app.models import DocumentRaw

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders_by_extension = self._build_loader_registry(loaders or default_loaders())

    def ingest_directory(self, source_dir: Path) -> IngestionResult:
        logger.info("Starting document ingestion from %s", source_dir)
        file_paths = scan_documents(source_dir, set(self._loaders_by_extension))
        logger.info("Detected %s supported document(s)", len(file_paths))

        documents: list[DocumentRaw] = []
        failed_files: list[Path] = []
        errors: list[str] = []

        for file_path in file_paths:
            loader = self._loaders_by_extension.get(file_path.suffix.lower())
            if loader is None:
                logger.warning("Skipping unsupported file: %s", file_path)
                continue

            try:
                documents.extend(loader.load(file_path))
            except Exception as exc:
                message = f"{file_path}: {exc}"
                failed_files.append(file_path)
                errors.append(message)
                logger.exception("Failed to ingest file: %s", file_path)

        logger.info("Finished document ingestion with %s extracted document unit(s)", len(documents))
        return IngestionResult(documents=documents, failed_files=failed_files, errors=errors)

    @staticmethod
    def _build_loader_registry(loaders: list[DocumentLoader]) -> dict[str, DocumentLoader]:
        registry: dict[str, DocumentLoader] = {}
        for loader in loaders:
            for extension in loader.supported_extensions:
                registry[extension.lower()] = loader
        return registry
