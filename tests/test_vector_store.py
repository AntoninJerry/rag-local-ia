from pathlib import Path

import pytest

from app.models import DocumentChunk
from app.vector_store.local_store import LocalJsonVectorStore, VectorStoreError


def make_chunk(chunk_id: str, text: str, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_file="guide.md",
        file_path=Path("data/documents/guide.md"),
        text=text,
        chunk_index=chunk_index,
        page_number=1,
    )


def test_vector_store_adds_persists_loads_and_searches_chunks(tmp_path: Path) -> None:
    store = LocalJsonVectorStore(storage_dir=tmp_path)
    chunks = [
        make_chunk("guide.md:1:0", "Premier chunk", 0),
        make_chunk("guide.md:1:1", "Deuxieme chunk", 1),
    ]

    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    reloaded_store = LocalJsonVectorStore(storage_dir=tmp_path)

    results = reloaded_store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "guide.md:1:0"
    assert results[0].text == "Premier chunk"
    assert results[0].source_file == "guide.md"
    assert results[0].file_path == Path("data/documents/guide.md")
    assert results[0].page_number == 1
    assert results[0].score == pytest.approx(1.0)
    assert (tmp_path / "index.json").exists()


def test_vector_store_returns_top_k_ordered_by_score(tmp_path: Path) -> None:
    store = LocalJsonVectorStore(storage_dir=tmp_path)
    store.add_chunks(
        [
            make_chunk("guide.md:1:0", "moins proche", 0),
            make_chunk("guide.md:1:1", "plus proche", 1),
            make_chunk("guide.md:1:2", "oppose", 2),
        ],
        [[0.7, 0.3], [1.0, 0.0], [-1.0, 0.0]],
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk_id for result in results] == ["guide.md:1:1", "guide.md:1:0"]
    assert results[0].score >= results[1].score


def test_vector_store_upserts_existing_chunk_id(tmp_path: Path) -> None:
    store = LocalJsonVectorStore(storage_dir=tmp_path)
    store.add_chunks([make_chunk("guide.md:1:0", "ancienne version")], [[1.0, 0.0]])
    store.add_chunks([make_chunk("guide.md:1:0", "nouvelle version")], [[0.0, 1.0]])

    results = store.search([0.0, 1.0], top_k=5)

    assert len(results) == 1
    assert results[0].text == "nouvelle version"


def test_vector_store_rejects_invalid_dimensions(tmp_path: Path) -> None:
    store = LocalJsonVectorStore(storage_dir=tmp_path)
    store.add_chunks([make_chunk("guide.md:1:0", "chunk")], [[1.0, 0.0]])

    with pytest.raises(VectorStoreError):
        store.add_chunks([make_chunk("guide.md:1:1", "autre chunk", 1)], [[1.0, 0.0, 0.0]])

    with pytest.raises(VectorStoreError):
        store.search([1.0, 0.0, 0.0], top_k=1)
