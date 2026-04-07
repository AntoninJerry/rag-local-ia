import logging
from pathlib import Path

from app.ingestion.loaders import PdfFileLoader, TextFileLoader
from app.ingestion.service import DocumentIngestionService


def test_text_file_loader_reads_txt_file_with_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Contenu texte", encoding="utf-8")

    documents = TextFileLoader().load(file_path)

    assert len(documents) == 1
    assert documents[0].source_file == "notes.txt"
    assert documents[0].file_path == file_path
    assert documents[0].page_number is None
    assert documents[0].text == "Contenu texte"


def test_ingestion_reads_txt_and_md_files(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "notes.txt").write_text("Texte TXT", encoding="utf-8")
    (documents_dir / "readme.md").write_text("# Titre MD", encoding="utf-8")
    (documents_dir / "ignored.docx").write_text("Ignore", encoding="utf-8")

    result = DocumentIngestionService().ingest_directory(documents_dir)

    assert len(result.documents) == 2
    assert {document.source_file for document in result.documents} == {"notes.txt", "readme.md"}
    assert result.failed_files == []
    assert result.errors == []


def test_ingestion_continues_when_one_file_fails(tmp_path: Path, caplog) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "ok.txt").write_text("Document valide", encoding="utf-8")
    (documents_dir / "broken.pdf").write_text("not a real pdf", encoding="utf-8")

    caplog.set_level(logging.ERROR)

    result = DocumentIngestionService().ingest_directory(documents_dir)

    assert len(result.documents) == 1
    assert result.documents[0].source_file == "ok.txt"
    assert result.failed_files == [documents_dir / "broken.pdf"]
    assert len(result.errors) == 1
    assert "Failed to ingest file" in caplog.text


def test_pdf_loader_extracts_text_page_by_page(monkeypatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, file_path: Path) -> None:
            self.pages = [FakePage("Page 1"), FakePage("Page 2")]

    monkeypatch.setattr("app.ingestion.loaders.PdfReader", FakeReader)

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF fake")

    documents = PdfFileLoader().load(file_path)

    assert [document.page_number for document in documents] == [1, 2]
    assert [document.text for document in documents] == ["Page 1", "Page 2"]
    assert all(document.source_file == "sample.pdf" for document in documents)
