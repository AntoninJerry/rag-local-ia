from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Local IA"
    app_env: str = "local"
    data_dir: Path = Path("data")
    documents_dir: Path = Path("data/documents")
    vector_store_dir: Path = Path("data/vector_store")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "local_stub"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
