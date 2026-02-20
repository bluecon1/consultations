from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    data_path: Path
    section_mapping_path: Path
    cache_path: Path
    llm_provider: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment: str
    azure_openai_api_key: str
    azure_openai_use_aad: bool
    azure_openai_managed_identity_client_id: str
    azure_openai_token_scope: str
    llm_timeout_seconds: int
    llm_max_retries: int
    prompt_excerpt_chars: int
    low_sample_threshold: int
    high_missingness_threshold: float
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float

    @classmethod
    def from_env(cls) -> "Settings":
        """Build validated runtime settings from environment variables.

        Input:
            Environment variables loaded via `.env` and process env.

        Output:
            A `Settings` instance with typed values and absolute local paths.

        Notes:
            Relative `DATA_CSV_PATH` and `CACHE_PATH` values are resolved
            against the project root (`neso-consultations/`).
        """
        data_path = Path(os.getenv("DATA_CSV_PATH", "data/data.csv"))
        if not data_path.is_absolute():
            data_path = ROOT_DIR / data_path

        cache_path = Path(os.getenv("CACHE_PATH", ".cache/summaries.sqlite"))
        if not cache_path.is_absolute():
            cache_path = ROOT_DIR / cache_path

        section_mapping_path = Path(
            os.getenv("SECTION_MAPPING_PATH", "data/survey questrion-section mapping.xlsx")
        )
        if not section_mapping_path.is_absolute():
            section_mapping_path = ROOT_DIR / section_mapping_path

        return cls(
            data_path=data_path,
            section_mapping_path=section_mapping_path,
            cache_path=cache_path,
            llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            azure_openai_use_aad=(os.getenv("AZURE_OPENAI_USE_AAD", "false").strip().lower() == "true"),
            azure_openai_managed_identity_client_id=os.getenv(
                "AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID", ""
            ).strip(),
            azure_openai_token_scope=os.getenv(
                "AZURE_OPENAI_TOKEN_SCOPE", "https://cognitiveservices.azure.com/.default"
            ).strip(),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "300")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            prompt_excerpt_chars=int(os.getenv("PROMPT_EXCERPT_CHARS", "280")),
            low_sample_threshold=int(os.getenv("LOW_SAMPLE_THRESHOLD", "8")),
            high_missingness_threshold=float(os.getenv("HIGH_MISSINGNESS_THRESHOLD", "0.35")),
            input_cost_per_1k_tokens=float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0008")),
            output_cost_per_1k_tokens=float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0032")),
        )

    @property
    def model_identity(self) -> str:
        """Return active model/deployment identity for cache keying."""
        if self.llm_provider == "azure":
            return self.azure_openai_deployment or self.openai_model
        return self.openai_model


def get_settings() -> Settings:
    """Return application settings for the current process.

    Output:
        A `Settings` object created from environment values.
    """
    return Settings.from_env()
