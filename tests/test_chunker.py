from pathlib import Path

from app.chunking.chunker import TextChunker
from app.models import DocumentRaw


def test_chunker_splits_document_into_chunks_with_metadata() -> None:
    document = DocumentRaw(
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        page_number=2,
        text=(
            "Premiere phrase complete. Deuxieme phrase complete. "
            "Troisieme phrase complete. Quatrieme phrase complete."
        ),
    )
    chunker = TextChunker(chunk_size=55, overlap=10)

    chunks = chunker.split(document)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "guide.md:2:0"
    assert chunks[0].source_file == "guide.md"
    assert chunks[0].file_path == Path("data/documents/guide.md")
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].text.endswith(".")


def test_chunker_returns_no_chunk_for_empty_text() -> None:
    document = DocumentRaw(
        source_file="empty.txt",
        file_path=Path("data/documents/empty.txt"),
        text="   \n   ",
    )

    assert TextChunker().split(document) == []
