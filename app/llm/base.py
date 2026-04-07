from typing import Protocol

from app.models import RetrievedChunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in retrieved document chunks."""
