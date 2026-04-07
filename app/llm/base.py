from typing import Protocol

from app.models import RetrievedChunk


class LLMClient(Protocol):
    def generate_answer(self, prompt: str) -> str:
        """Generate an answer from a fully built prompt."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in retrieved document chunks."""
