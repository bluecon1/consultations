from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request

from neso_consultations.llm.base import LLMJsonResult, LLMProvider
from neso_consultations.models import LLMUsage


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: int = 300,
        max_retries: int = 2,
    ) -> None:
        """Initialise OpenAI REST client settings.

        Inputs:
            api_key: OpenAI API key.
            model: Model identifier used for chat completions.
            base_url: Optional API base URL (defaults to public OpenAI endpoint).
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._timeout_seconds = max(30, int(timeout_seconds))
        self._max_retries = max(0, int(max_retries))

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> LLMJsonResult:
        """Run a JSON-constrained chat completion call.

        Inputs:
            system_prompt: Global instructions for summarisation behavior.
            user_prompt: Request payload with source consultation evidence.
            temperature: Sampling control.

        Output:
            `LLMJsonResult` containing parsed JSON payload and token usage.
        """
        payload = {
            "model": self._model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response_json = self._post_json("/chat/completions", payload)

        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI response did not contain choices.")

        content = choices[0].get("message", {}).get("content", "{}")
        parsed_payload = _safe_json_loads(content)

        usage = response_json.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        return LLMJsonResult(
            payload=parsed_payload,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an authenticated POST request and return parsed JSON object.

        Inputs:
            path: Relative API path, e.g. `/chat/completions`.
            payload: JSON-serialisable request body.

        Output:
            Parsed dictionary response.

        Raises:
            RuntimeError for HTTP/network/JSON-shape failures.
        """
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except TimeoutError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise RuntimeError(
                        f"OpenAI request timed out after {self._timeout_seconds}s "
                        f"(attempts={self._max_retries + 1})."
                    ) from exc
                time.sleep(1.5 * (attempt + 1))
            except error.HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="ignore")
                # Retry transient status codes.
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self._max_retries:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI HTTP error {exc.code}: {raw_error}") from exc
            except error.URLError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise RuntimeError(f"OpenAI network error: {exc.reason}") from exc
                time.sleep(1.5 * (attempt + 1))
        else:
            # Defensive: loop always breaks or raises; this is for type-safety.
            raise RuntimeError(f"OpenAI request failed: {last_error}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI response was not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI response JSON was not an object.")

        return parsed


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Best-effort parser for model output that should be a JSON object.

    Input:
        text: Raw text content from model response.

    Output:
        Parsed dictionary, or empty dictionary when parsing fails.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

    return {}
