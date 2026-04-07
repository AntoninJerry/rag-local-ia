from pathlib import Path

from pydantic import BaseModel, Field

from app.models import IndexingResult, QueryRequest, QueryResponse


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class IndexRequest(BaseModel):
    source_path: Path = Field(..., description="Dossier local contenant les documents a indexer.")


IndexResponse = IndexingResult
AskRequest = QueryRequest
AskResponse = QueryResponse
