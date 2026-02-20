from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from neso_consultations.models import LLMUsage


@dataclass(frozen=True)
class LLMJsonResult:
    payload: dict[str, Any]
    usage: LLMUsage


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> LLMJsonResult:
        """Execute one model call and return structured JSON plus token usage.

        Inputs:
            system_prompt: Instructional policy/format constraints.
            user_prompt: Request payload and source evidence text.
            temperature: Sampling temperature for generation.

        Output:
            `LLMJsonResult` with parsed dictionary payload and token counts.
        """
        raise NotImplementedError
