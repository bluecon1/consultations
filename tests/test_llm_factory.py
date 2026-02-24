from pathlib import Path

import pytest

from neso_consultations.config import Settings
from neso_consultations.llm.factory import build_llm_provider
from neso_consultations.llm.noop_provider import NoOpLLMProvider


def _settings(*, llm_provider: str) -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        data_path=root / "data" / "data.csv",
        section_mapping_path=root / "data" / "survey questrion-section mapping.xlsx",
        cache_path=root / ".cache" / "test.sqlite",
        cache_enabled=True,
        llm_provider=llm_provider,
        openai_api_key="test",
        openai_model="gpt-4.1-mini",
        openai_base_url=None,
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_version="2024-06-01",
        azure_openai_deployment="gpt-4.1-mini",
        azure_openai_api_key="test",
        azure_openai_use_aad=False,
        azure_openai_managed_identity_client_id="",
        azure_openai_token_scope="https://cognitiveservices.azure.com/.default",
        llm_timeout_seconds=60,
        llm_max_retries=1,
        prompt_excerpt_chars=280,
        low_sample_threshold=8,
        high_missingness_threshold=0.35,
        input_cost_per_1k_tokens=0.0008,
        output_cost_per_1k_tokens=0.0032,
    )


def test_factory_returns_noop_when_llm_not_required() -> None:
    provider = build_llm_provider(_settings(llm_provider="openai"), require_llm=False)
    assert isinstance(provider, NoOpLLMProvider)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        build_llm_provider(_settings(llm_provider="unknown"), require_llm=True)
