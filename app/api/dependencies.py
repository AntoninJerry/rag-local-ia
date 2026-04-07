from app.chunking.chunker import TextChunker
from app.core.config import settings
from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider
from app.indexing.service import IndexingService
from app.ingestion.service import DocumentIngestionService
from app.llm.client import create_llm_client
from app.llm.prompt_builder import RagPromptBuilder
from app.querying.service import QueryService
from app.rag.pipeline import RagPipeline
from app.retrieval.retriever import Retriever
from app.vector_store.local_store import LocalJsonVectorStore


def get_indexing_service() -> IndexingService:
    vector_store = LocalJsonVectorStore(storage_dir=settings.vector_store_dir)
    return IndexingService(
        ingestion_service=DocumentIngestionService(),
        chunker=TextChunker.from_settings(settings),
        embedding_provider=SentenceTransformersEmbeddingProvider(settings.embedding_model_name),
        vector_store=vector_store,
    )


def get_query_service() -> QueryService:
    embedding_provider = SentenceTransformersEmbeddingProvider(settings.embedding_model_name)
    vector_store = LocalJsonVectorStore(storage_dir=settings.vector_store_dir)
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=settings.retrieval_top_k,
    )
    pipeline = RagPipeline(
        retriever=retriever,
        llm_client=create_llm_client(settings),
        prompt_builder=RagPromptBuilder(),
        default_top_k=settings.retrieval_top_k,
    )
    return QueryService(pipeline=pipeline)
