"""
LLM Client
====================
Lightweight DeepSeek API client built on the OpenAI SDK.

DeepSeek uses an OpenAI-compatible API, so we reuse the ``openai``
package with a different ``base_url``.

Features:
    - Single and batch chat completions
    - JSON mode for structured output
    - Configurable temperature, max_tokens, timeout
    - Instance-level caching (one client per config)
"""

from __future__ import annotations

import json
from typing import Any

from ..config import LLMConfig


class LLMClient:
    """LLM API client.

    Args:
        config: An ``LLMConfig`` with API key and settings.

    Example::

        from ragforge import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(api_key="sk-..."))
        response = client.chat("What is RAG?")
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is not None:
            return self._client
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout,
        )
        return self._client

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a single chat completion request.

        Args:
            prompt: User message.
            system: Optional system prompt.
            temperature: Override config temperature.

        Returns:
            Assistant response text.
        """
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=temperature or self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict | list:
        """Send a chat request expecting JSON output.

        Args:
            prompt: User message.
            system: Optional system prompt.
            temperature: Override config temperature.

        Returns:
            Parsed JSON object (dict or list).
        """
        raw = self.chat(prompt, system=system, temperature=temperature)
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def chat_batch(
        self,
        prompts: list[str],
        system: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Send multiple chat requests sequentially.

        Args:
            prompts: List of user messages.
            system: Optional system prompt (shared).
            temperature: Override config temperature.

        Returns:
            List of assistant response texts.
        """
        return [
            self.chat(p, system=system, temperature=temperature)
            for p in prompts
        ]
