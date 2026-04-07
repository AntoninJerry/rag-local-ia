from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import settings


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class IndexRequest(BaseModel):
    source_path: Path = Field(..., description="Dossier local contenant les documents a indexer.")


class IndexResponse(BaseModel):
    indexed_documents: int
    source_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.retrieval_top_k, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
