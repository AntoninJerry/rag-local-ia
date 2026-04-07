from typing import Protocol

from app.ingestion.models import DocumentChunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: list[DocumentChunk]) -> str:
        """Generate an answer grounded in retrieved document chunks."""
