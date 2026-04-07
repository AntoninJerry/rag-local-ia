from app.models import RetrievedChunk


class Retriever:
    """Coordinates vector search for a user question."""

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        # TODO: Connect embeddings and vector store search.
        raise NotImplementedError("Retriever is not connected yet.")
