from __future__ import annotations

from neso_consultations.config import Settings
from neso_consultations.llm.base import LLMProvider
from neso_consultations.llm.noop_provider import NoOpLLMProvider


def build_llm_provider(settings: Settings, *, require_llm: bool = True) -> LLMProvider:
    """Create an LLM provider from environment-backed settings.

    Inputs:
        settings: Runtime configuration object.
        require_llm: When false, returns a no-op provider for list/read commands.

    Output:
        Concrete provider implementing `LLMProvider`.

    Raises:
        ValueError for unsupported provider names or missing provider credentials.
    """
    if not require_llm:
        return NoOpLLMProvider()

    provider_name = settings.llm_provider.lower()
    if provider_name == "openai":
        from neso_consultations.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider_name == "azure":
        from neso_consultations.llm.azure_openai_provider import AzureOpenAIProvider

        return AzureOpenAIProvider(
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key,
            use_aad=settings.azure_openai_use_aad,
            managed_identity_client_id=settings.azure_openai_managed_identity_client_id,
            token_scope=settings.azure_openai_token_scope,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
