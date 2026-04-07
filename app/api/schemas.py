from pathlib import Path

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class IndexRequest(BaseModel):
    source_path: Path = Field(..., description="Dossier local contenant les documents a indexer.")


class IndexResponse(BaseModel):
    indexed_documents: int
    source_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
