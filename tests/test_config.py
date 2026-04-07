import pytest
from pydantic import ValidationError

from app.chunking.chunker import TextChunker
from app.core.config import Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings()

    assert settings.documents_dir.as_posix() == "data/documents"
    assert settings.vector_store_dir.as_posix() == "data/vector_store"
    assert settings.embedding_model_name
    assert settings.llm_provider == "local_stub"
    assert settings.llm_model_name == "local-stub"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.retrieval_top_k == 5
    assert settings.log_level == "INFO"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_chunker_can_be_created_from_settings() -> None:
    settings = Settings(chunk_size=500, chunk_overlap=50)

    chunker = TextChunker.from_settings(settings)

    assert chunker.chunk_size == 500
    assert chunker.overlap == 50
