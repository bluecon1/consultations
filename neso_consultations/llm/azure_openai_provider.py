from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, parse, request

from neso_consultations.llm.base import LLMJsonResult, LLMProvider
from neso_consultations.models import LLMUsage


class AzureOpenAIProvider(LLMProvider):
    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_version: str,
        api_key: str = "",
        use_aad: bool = False,
        managed_identity_client_id: str = "",
        token_scope: str = "https://cognitiveservices.azure.com/.default",
        timeout_seconds: int = 300,
        max_retries: int = 2,
    ) -> None:
        """Initialise Azure OpenAI provider settings.

        Inputs:
            endpoint: Azure OpenAI resource endpoint.
            deployment: Deployment name to call.
            api_version: API version query parameter.
            api_key: API key (used when `use_aad=False`).
            use_aad: If true, uses `DefaultAzureCredential` for bearer tokens.
            managed_identity_client_id: Optional client ID for user-assigned
                managed identity when using AAD auth.
            token_scope: OAuth scope for Azure OpenAI.
            timeout_seconds: Request timeout per attempt.
            max_retries: Retry attempts for transient failures.
        """
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is not set.")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION is not set.")
        if not use_aad and not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set (or enable AZURE_OPENAI_USE_AAD=true).")

        self._endpoint = endpoint.rstrip("/")
        self._deployment = deployment
        self._api_version = api_version
        self._api_key = api_key
        self._use_aad = use_aad
        self._managed_identity_client_id = managed_identity_client_id
        self._token_scope = token_scope
        self._timeout_seconds = max(30, int(timeout_seconds))
        self._max_retries = max(0, int(max_retries))

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> LLMJsonResult:
        """Run one Azure OpenAI chat completion request in JSON mode."""
        payload = {
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response_json = self._post_json(payload)

        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError("Azure OpenAI response did not contain choices.")

        content = choices[0].get("message", {}).get("content", "{}")
        parsed_payload = _safe_json_loads(content)

        usage = response_json.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        return LLMJsonResult(
            payload=parsed_payload,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = f"/openai/deployments/{self._deployment}/chat/completions"
        query = parse.urlencode({"api-version": self._api_version})
        url = f"{self._endpoint}{path}?{query}"

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._use_aad:
            headers["Authorization"] = f"Bearer {self._get_aad_token()}"
        else:
            headers["api-key"] = self._api_key

        req = request.Request(url=url, data=body, method="POST", headers=headers)

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
                        f"Azure OpenAI request timed out after {self._timeout_seconds}s "
                        f"(attempts={self._max_retries + 1})."
                    ) from exc
                time.sleep(1.5 * (attempt + 1))
            except error.HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="ignore")
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self._max_retries:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Azure OpenAI HTTP error {exc.code}: {raw_error}") from exc
            except error.URLError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise RuntimeError(f"Azure OpenAI network error: {exc.reason}") from exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Azure OpenAI request failed: {last_error}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Azure OpenAI response was not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("Azure OpenAI response JSON was not an object.")

        return parsed

    def _get_aad_token(self) -> str:
        """Acquire AAD token using DefaultAzureCredential when enabled."""
        try:
            from azure.identity import DefaultAzureCredential
        except Exception as exc:
            raise RuntimeError(
                "azure-identity is required for AAD auth. Install it or use API key auth."
            ) from exc

        if self._managed_identity_client_id:
            credential = DefaultAzureCredential(
                managed_identity_client_id=self._managed_identity_client_id
            )
        else:
            credential = DefaultAzureCredential()
        token = credential.get_token(self._token_scope)
        return token.token


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Best-effort parser for model output that should be a JSON object."""
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
