from pathlib import Path

import typer

from app.core.config import settings

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
    # TODO: Wire the ingestion, chunking, embedding and vector store pipeline.
    typer.echo(f"Indexation non connectee pour le moment: {path}")


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
    # TODO: Wire retrieval and local LLM generation.
    typer.echo(f"Question recue: {question}")
    typer.echo(f"Top k retrieval: {top_k}")
    typer.echo("Le moteur RAG n'est pas encore connecte.")


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


def main() -> None:
    cli()
