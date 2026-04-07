from pathlib import Path

import pytest

from app.llm.prompt_builder import RagPromptBuilder
from app.models import RetrievedChunk


def make_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="README.md:1:0",
        source_file="README.md",
        file_path=Path("data/documents/README.md"),
        page_number=1,
        chunk_index=0,
        text="Le projet construit un assistant RAG local.",
        score=0.91,
    )


def test_prompt_builder_includes_question_context_and_sources() -> None:
    prompt = RagPromptBuilder().build(
        question="Quel est l'objectif du projet ?",
        retrieved_chunks=[make_retrieved_chunk()],
    )

    assert "Quel est l'objectif du projet ?" in prompt
    assert "Le projet construit un assistant RAG local." in prompt
    assert "source_id: README.md:1:0" in prompt
    assert "Fichier: README.md, page 1" in prompt
    assert "Score: 0.9100" in prompt
    assert "Sources utilisees" in prompt
    assert "Si le contexte ne contient pas l'information demandee" in prompt


def test_prompt_builder_handles_empty_context() -> None:
    prompt = RagPromptBuilder().build("Question sans contexte ?", [])

    assert "Aucun extrait documentaire n'a ete fourni." in prompt
    assert "dis clairement que tu ne sais pas" in prompt


def test_prompt_builder_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        RagPromptBuilder().build("   ", [])
