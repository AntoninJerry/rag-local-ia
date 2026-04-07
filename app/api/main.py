from fastapi import FastAPI

from app.api.schemas import AskRequest, AskResponse, HealthResponse, IndexRequest, IndexResponse
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, environment=settings.app_env)


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    # TODO: Wire the ingestion, chunking, embedding and vector store pipeline.
    return IndexResponse(source_path=request.source_path)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    # TODO: Wire retrieval and local LLM generation.
    return AskResponse(answer="Le moteur RAG n'est pas encore connecte.", sources=[])
