import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings, settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an LLM client cannot generate an answer."""


@dataclass(frozen=True)
class LLMClientConfig:
    provider: str
    model_name: str
    timeout_seconds: float

    @classmethod
    def from_settings(cls, config: Settings = settings) -> "LLMClientConfig":
        return cls(
            provider=config.llm_provider,
            model_name=config.llm_model_name,
            timeout_seconds=config.llm_timeout_seconds,
        )


class LocalStubLLMClient(LLMClient):
    """Deterministic local client used while real LLM backends are not wired.

    To add Ollama, OpenRouter or another backend later, create another class
    implementing LLMClient.generate_answer(prompt), then register it in
    create_llm_client without changing the retrieval or prompt-building pipeline.
    """

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def generate_answer(self, prompt: str) -> str:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise LLMClientError("Prompt must not be empty.")

        logger.info(
            "Generating answer with provider=%s model=%s",
            self.config.provider,
            self.config.model_name,
        )
        return (
            "Reponse non generee par un vrai LLM pour le moment. "
            "Le prompt RAG a ete construit et transmis au client local_stub."
        )


LLMClientFactory = Callable[[LLMClientConfig], LLMClient]


def create_llm_client(
    config: Settings = settings,
    factories: dict[str, LLMClientFactory] | None = None,
) -> LLMClient:
    client_config = LLMClientConfig.from_settings(config)
    registry = factories or {"local_stub": LocalStubLLMClient}
    try:
        factory = registry[client_config.provider]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise LLMClientError(
            f"Unsupported LLM provider '{client_config.provider}'. Supported providers: {supported}"
        ) from exc

    try:
        return factory(client_config)
    except Exception as exc:
        raise LLMClientError(f"Failed to initialize LLM provider '{client_config.provider}': {exc}") from exc
