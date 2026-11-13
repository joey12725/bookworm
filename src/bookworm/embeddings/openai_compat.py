"""Embeddings from any OpenAI-compatible ``/embeddings`` endpoint.

Covers OpenAI, OpenRouter, Ollama, LM Studio, vLLM, llama.cpp server and friends.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from bookworm.embeddings.base import l2_normalize
from bookworm.errors import ProviderError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int = 4,
    backoff: float = 1.0,
) -> dict[str, Any]:
    """POST with exponential backoff on rate limits, server errors and timeouts."""
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=payload)
        except httpx.TransportError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code < 400:
                data = response.json()
                if isinstance(data, dict) and "error" in data and len(data) == 1:
                    raise ProviderError(f"{url}: {data['error']}")
                if not isinstance(data, dict):
                    raise ProviderError(f"{url}: unexpected response")
                return data
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRY_STATUS:
                break
        if attempt < max_retries:
            time.sleep(backoff * 2**attempt)
    raise ProviderError(f"Request to {url} failed: {last_error}")


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff = backoff
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)
        self._client.headers.update(headers)

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def _request(self, batch: Sequence[str]) -> list[list[float]]:
        data = post_json(
            self._client,
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": list(batch)},
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise ProviderError(
                f"Embedding response had {len(items or [])} vectors for {len(batch)} inputs"
            )
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [l2_normalize(item["embedding"]) for item in items]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]
