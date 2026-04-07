import pytest

from app.core.config import Settings
from app.llm.client import LLMClientConfig, LLMClientError, LocalStubLLMClient, create_llm_client


def test_llm_client_config_is_created_from_settings() -> None:
    settings = Settings(
        llm_provider="local_stub",
        llm_model_name="test-model",
        llm_timeout_seconds=12.5,
    )

    config = LLMClientConfig.from_settings(settings)

    assert config.provider == "local_stub"
    assert config.model_name == "test-model"
    assert config.timeout_seconds == 12.5


def test_local_stub_llm_client_generates_deterministic_answer() -> None:
    client = LocalStubLLMClient(
        LLMClientConfig(provider="local_stub", model_name="local-stub", timeout_seconds=30)
    )

    answer = client.generate_answer("Prompt RAG")

    assert "local_stub" in answer
    assert "Prompt RAG" not in answer


def test_local_stub_llm_client_rejects_empty_prompt() -> None:
    client = LocalStubLLMClient(
        LLMClientConfig(provider="local_stub", model_name="local-stub", timeout_seconds=30)
    )

    with pytest.raises(LLMClientError):
        client.generate_answer("   ")


def test_create_llm_client_uses_provider_from_settings() -> None:
    settings = Settings(llm_provider="local_stub")

    client = create_llm_client(settings)

    assert isinstance(client, LocalStubLLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    settings = Settings(llm_provider="unknown")

    with pytest.raises(LLMClientError):
        create_llm_client(settings)
