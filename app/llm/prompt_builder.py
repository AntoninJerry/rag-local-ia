from app.models import RetrievedChunk

DEFAULT_SYSTEM_INSTRUCTIONS = """Tu es un assistant de recherche local.
Tu dois repondre uniquement a partir du contexte documentaire fourni.
Si le contexte ne contient pas l'information demandee, dis clairement que tu ne sais pas.
N'invente pas de faits, de chiffres, de noms ou de sources.
Reponds de facon concise, fiable et tracable."""

DEFAULT_RESPONSE_INSTRUCTIONS = """Format attendu :
1. Reponse concise
2. Sources utilisees

Dans la section sources utilisees, cite les identifiants d'extraits pertinents."""


class RagPromptBuilder:
    def __init__(
        self,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        response_instructions: str = DEFAULT_RESPONSE_INSTRUCTIONS,
    ) -> None:
        self.system_instructions = system_instructions.strip()
        self.response_instructions = response_instructions.strip()

    def build(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        context = self._format_context(retrieved_chunks)
        return "\n\n".join(
            [
                "# Instructions",
                self.system_instructions,
                "# Question utilisateur",
                cleaned_question,
                "# Contexte documentaire",
                context,
                "# Consignes de reponse",
                self.response_instructions,
            ]
        )

    def _format_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "Aucun extrait documentaire n'a ete fourni."

        return "\n\n".join(self._format_chunk(chunk) for chunk in retrieved_chunks)

    @staticmethod
    def _format_chunk(chunk: RetrievedChunk) -> str:
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        return (
            f"[source_id: {chunk.chunk_id}]\n"
            f"Fichier: {chunk.source_file}{page}\n"
            f"Chemin: {chunk.file_path}\n"
            f"Score: {chunk.score:.4f}\n"
            f"Extrait:\n{chunk.text.strip()}"
        )
