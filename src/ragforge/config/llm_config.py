"""
LLM Configuration
=================
Configuration for LLM backends (DeepSeek, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM API access.

    Attributes:
        api_key:
            API key for the LLM service.
        base_url:
            Base URL of the LLM API endpoint.
            Defaults to DeepSeek API.
        model:
            Model name to use.
            Defaults to ``deepseek-chat``.
        temperature:
            Sampling temperature for generation (0.0 = deterministic).
        max_tokens:
            Maximum number of tokens in the response.
        timeout:
            Request timeout in seconds.
    """

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 60
