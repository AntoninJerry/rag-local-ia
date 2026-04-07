import logging

from fastapi import Depends, FastAPI, HTTPException

from app.api.dependencies import get_indexing_service, get_query_service
from app.api.schemas import AskRequest, AskResponse, HealthResponse, IndexRequest, IndexResponse
from app.core.config import settings
from app.core.logging import configure_logging
from app.indexing.service import IndexingService
from app.querying.service import QueryService

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, environment=settings.app_env)


@app.post("/index", response_model=IndexResponse)
def index_documents(
    request: IndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    try:
        result = indexing_service.index_directory(request.source_path)
        return IndexResponse.model_validate(result.model_dump())
    except ValueError as exc:
        logger.warning("Invalid index request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Index endpoint failed")
        raise HTTPException(status_code=500, detail="Indexing failed.") from exc


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    query_service: QueryService = Depends(get_query_service),
) -> AskResponse:
    try:
        response = query_service.ask(request)
        return AskResponse.model_validate(response.model_dump())
    except ValueError as exc:
        logger.warning("Invalid ask request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ask endpoint failed")
        raise HTTPException(status_code=500, detail="Query failed.") from exc
