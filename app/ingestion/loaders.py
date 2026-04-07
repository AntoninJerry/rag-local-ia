from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from app.models import DocumentRaw


class DocumentLoader(Protocol):
    supported_extensions: set[str]

    def load(self, file_path: Path) -> list[DocumentRaw]:
        """Extract text from a supported file."""


class TextFileLoader:
    supported_extensions = {".txt", ".md"}

    def load(self, file_path: Path) -> list[DocumentRaw]:
        text = file_path.read_text(encoding="utf-8")
        return [
            DocumentRaw(
                source_file=file_path.name,
                file_path=file_path,
                text=text,
            )
        ]


class PdfFileLoader:
    supported_extensions = {".pdf"}

    def load(self, file_path: Path) -> list[DocumentRaw]:
        reader = PdfReader(file_path)
        documents: list[DocumentRaw] = []

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append(
                DocumentRaw(
                    source_file=file_path.name,
                    file_path=file_path,
                    page_number=page_index,
                    text=text,
                )
            )

        return documents


def default_loaders() -> list[DocumentLoader]:
    # TODO: Add a DOCX loader here when DOCX support becomes part of the MVP scope.
    return [TextFileLoader(), PdfFileLoader()]
