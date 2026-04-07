from app.ingestion.models import Document, DocumentChunk


class TextChunker:
    """Split documents into overlapping character chunks for the MVP."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be positive and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[DocumentChunk]:
        # TODO: Replace with token-aware chunking if the selected embedding model requires it.
        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(document.text):
            end = start + self.chunk_size
            chunk_text = document.text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source_path=document.source_path,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
            start += self.chunk_size - self.overlap

        return chunks
