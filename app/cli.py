from pathlib import Path

import typer

from app.chunking.chunker import TextChunker
from app.core.config import settings
from app.core.logging import configure_logging
from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider
from app.indexing.service import IndexingService
from app.ingestion.service import DocumentIngestionService
from app.llm.client import create_llm_client
from app.llm.prompt_builder import RagPromptBuilder
from app.models import IndexingResult, QueryRequest, QueryResponse
from app.querying.service import QueryService
from app.rag.pipeline import RagPipeline
from app.retrieval.retriever import Retriever
from app.vector_store.local_store import LocalJsonVectorStore

configure_logging()

cli = typer.Typer(help="CLI locale pour l'assistant RAG.")


@cli.callback()
def callback() -> None:
    """Assistant RAG local."""


@cli.command()
def index(
    path: Path = typer.Argument(
        settings.documents_dir,
        help="Dossier contenant les documents a indexer.",
    )
) -> None:
    if not path.exists():
        typer.secho(f"Erreur: le dossier n'existe pas: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.secho(f"Erreur: le chemin n'est pas un dossier: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Indexation du dossier: {path}")
    result = build_indexing_service().index_directory(path)
    _print_indexing_result(result)
    if result.errors:
        raise typer.Exit(code=1)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Question a poser aux documents indexes."),
    top_k: int = typer.Option(
        settings.retrieval_top_k,
        min=1,
        max=50,
        help="Nombre de chunks a recuperer.",
    ),
) -> None:
    try:
        response = build_query_service().ask(QueryRequest(question=question, top_k=top_k))
    except Exception as exc:
        typer.secho(f"Erreur: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    _print_query_response(response)


@cli.command()
def info() -> None:
    typer.echo(f"Application: {settings.app_name}")
    typer.echo(f"Environnement: {settings.app_env}")
    typer.echo(f"Dossier documents: {settings.documents_dir}")
    typer.echo(f"Dossier vector store: {settings.vector_store_dir}")
    typer.echo(f"Modele embeddings: {settings.embedding_model_name}")
    typer.echo(f"Fournisseur LLM: {settings.llm_provider}")
    typer.echo(f"Modele LLM: {settings.llm_model_name}")
    typer.echo(f"Timeout LLM: {settings.llm_timeout_seconds}s")
    typer.echo(f"Chunk size: {settings.chunk_size}")
    typer.echo(f"Chunk overlap: {settings.chunk_overlap}")
    typer.echo(f"Top k retrieval: {settings.retrieval_top_k}")
    typer.echo(f"Niveau logs: {settings.log_level}")


def build_indexing_service() -> IndexingService:
    vector_store = LocalJsonVectorStore(storage_dir=settings.vector_store_dir)
    return IndexingService(
        ingestion_service=DocumentIngestionService(),
        chunker=TextChunker.from_settings(settings),
        embedding_provider=SentenceTransformersEmbeddingProvider(settings.embedding_model_name),
        vector_store=vector_store,
    )


def build_query_service() -> QueryService:
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


def _print_indexing_result(result: IndexingResult) -> None:
    typer.echo("")
    typer.echo("Resume d'indexation")
    typer.echo(f"- Fichiers lus: {result.files_read}")
    typer.echo(f"- Fichiers ignores/en erreur: {result.files_ignored}")
    typer.echo(f"- Documents extraits: {result.indexed_documents}")
    typer.echo(f"- Chunks crees: {result.indexed_chunks}")
    typer.echo(f"- Duree: {result.duration_seconds:.2f}s")

    if result.errors:
        typer.secho("- Erreurs:", fg=typer.colors.YELLOW)
        for error in result.errors:
            typer.echo(f"  - {error}")


def _print_query_response(response: QueryResponse) -> None:
    typer.echo("")
    typer.echo("Reponse")
    typer.echo(response.answer)
    typer.echo("")
    typer.echo("Sources utilisees")
    if not response.sources:
        typer.echo("- Aucune source pertinente trouvee.")
        return

    for source in response.sources:
        page = f", page {source.page_number}" if source.page_number is not None else ""
        typer.echo(f"- {source.source_file}{page} | chunk={source.chunk_id} | score={source.score:.4f}")
        typer.echo(f"  chemin: {source.file_path}")
        typer.echo(f"  extrait: {source.text[:240]}")


def main() -> None:
    cli()
