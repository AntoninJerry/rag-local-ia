from pathlib import Path

from typer.testing import CliRunner

from app import cli as cli_module
from app.models import IndexingResult, QueryResponse, RetrievedChunk

runner = CliRunner()


class FakeIndexingService:
    def index_directory(self, source_dir: Path) -> IndexingResult:
        return IndexingResult(
            source_path=source_dir,
            files_read=2,
            files_ignored=1,
            indexed_documents=2,
            indexed_chunks=4,
            duration_seconds=1.25,
        )


class FakeQueryService:
    def ask(self, request) -> QueryResponse:
        return QueryResponse(
            answer=f"Reponse pour: {request.question}",
            sources=[
                RetrievedChunk(
                    chunk_id="guide.md:0:0",
                    source_file="guide.md",
                    file_path=Path("data/documents/guide.md"),
                    text="Extrait pertinent pour la question.",
                    chunk_index=0,
                    score=0.95,
                )
            ],
        )


def test_cli_index_displays_indexing_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli_module, "build_indexing_service", lambda: FakeIndexingService())

    result = runner.invoke(cli_module.cli, ["index", str(tmp_path)])

    assert result.exit_code == 0
    assert "Resume d'indexation" in result.output
    assert "Fichiers lus: 2" in result.output
    assert "Chunks crees: 4" in result.output


def test_cli_ask_displays_answer_and_sources(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "build_query_service", lambda: FakeQueryService())

    result = runner.invoke(cli_module.cli, ["ask", "Question test", "--top-k", "1"])

    assert result.exit_code == 0
    assert "Reponse pour: Question test" in result.output
    assert "Sources utilisees" in result.output
    assert "guide.md" in result.output
    assert "score=0.9500" in result.output


def test_cli_index_rejects_missing_directory() -> None:
    result = runner.invoke(cli_module.cli, ["index", "missing-directory"])

    assert result.exit_code == 1
    assert "le dossier n'existe pas" in result.output
