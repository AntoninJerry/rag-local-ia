from pathlib import Path

import typer

from app.core.config import settings

cli = typer.Typer(help="CLI locale pour l'assistant RAG.")


@cli.callback()
def callback() -> None:
    """Assistant RAG local."""


@cli.command()
def index(path: Path = typer.Argument(..., help="Dossier contenant les documents a indexer.")) -> None:
    # TODO: Wire the ingestion, chunking, embedding and vector store pipeline.
    typer.echo(f"Indexation non connectee pour le moment: {path}")


@cli.command()
def ask(question: str = typer.Argument(..., help="Question a poser aux documents indexes.")) -> None:
    # TODO: Wire retrieval and local LLM generation.
    typer.echo(f"Question recue: {question}")
    typer.echo("Le moteur RAG n'est pas encore connecte.")


@cli.command()
def info() -> None:
    typer.echo(f"Application: {settings.app_name}")
    typer.echo(f"Dossier de donnees: {settings.data_dir}")


def main() -> None:
    cli()
