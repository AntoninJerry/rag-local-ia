from app.core.config import Settings, settings
from app.models import DocumentChunk, DocumentRaw

SPLIT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


class TextChunker:
    """Split documents into overlapping character chunks for the MVP."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be positive and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @classmethod
    def from_settings(cls, config: Settings = settings) -> "TextChunker":
        return cls(chunk_size=config.chunk_size, overlap=config.chunk_overlap)

    def split(self, document: DocumentRaw) -> list[DocumentChunk]:
        # TODO: Replace with token-aware chunking if the selected embedding model requires it.
        text = self._normalize_text(document.text)
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = self._find_chunk_end(text, start)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.source_file}:{document.page_number or 0}:{chunk_index}",
                        text=chunk_text,
                        source_file=document.source_file,
                        file_path=document.file_path,
                        page_number=document.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            start = max(end - self.overlap, start + 1)
            while start < len(text) and text[start].isspace():
                start += 1

        return chunks

    def _find_chunk_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.chunk_size, len(text))
        if hard_end >= len(text):
            return len(text)

        min_end = start + max(self.chunk_size // 2, 1)
        for separator in SPLIT_SEPARATORS:
            split_at = text.rfind(separator, start, hard_end)
            if split_at >= min_end:
                return split_at + len(separator)

        return hard_end

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()
