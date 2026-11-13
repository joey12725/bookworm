"""Chat models for grounded answers. Any OpenAI-compatible ``/chat/completions`` endpoint."""

from __future__ import annotations

from typing import Protocol

import httpx

from bookworm.config import PROVIDERS, Settings
from bookworm.embeddings.openai_compat import post_json
from bookworm.errors import BookwormError, ProviderError

Message = dict[str, str]


class ChatModel(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, messages: list[Message]) -> str: ...


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = client or httpx.Client(timeout=timeout)
        if api_key:
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def name(self) -> str:
        return self.model

    def complete(self, messages: list[Message]) -> str:
        data = post_json(
            self._client,
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": self.temperature},
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Chat response had no message content") from exc
        if not content:
            raise ProviderError("Chat model returned an empty answer")
        return str(content).strip()


def create_chat_model(settings: Settings) -> ChatModel | None:
    """The configured chat model, or None for extractive answers."""
    provider = settings.chat_provider
    if not provider:
        return None
    preset = PROVIDERS.get(provider, {})
    base_url = settings.chat_base_url or preset.get("base_url")
    model = settings.chat_model or preset.get("chat_model")
    if not base_url or not model:
        raise BookwormError(
            f"Chat provider {provider!r} needs BOOKWORM_CHAT_BASE_URL and BOOKWORM_CHAT_MODEL"
        )
    if preset.get("key_env") and not settings.chat_api_key:
        raise BookwormError(
            f"Chat provider {provider!r} needs an API key: set {preset['key_env']} "
            "or BOOKWORM_CHAT_API_KEY"
        )
    return OpenAICompatibleChat(base_url=base_url, model=model, api_key=settings.chat_api_key)
