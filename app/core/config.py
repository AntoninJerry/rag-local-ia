from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Local IA"
    app_env: str = "local"
    data_dir: Path = Path("data")
    documents_dir: Path = Path("data/documents")
    vector_store_dir: Path = Path("data/vector_store")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "local_stub"
    llm_model_name: str = "local-stub"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
