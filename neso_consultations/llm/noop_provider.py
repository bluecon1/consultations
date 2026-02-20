from __future__ import annotations

from neso_consultations.llm.base import LLMJsonResult, LLMProvider
from neso_consultations.models import LLMUsage


class NoOpLLMProvider(LLMProvider):
    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> LLMJsonResult:
        """Raise a clear error when summary generation is requested without an LLM.

        Inputs:
            system_prompt/user_prompt/temperature are accepted to match interface.

        Output:
            No return value; always raises `RuntimeError`.
        """
        raise RuntimeError("LLM provider is not configured. Set OPENAI_API_KEY to generate summaries.")
