from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import httpx

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class EmbeddingService:
    def __init__(self, *, api_key: str, model: str = DEFAULT_EMBEDDING_MODEL, timeout: int = 60) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, texts: list[str], *, max_retries: int = 3) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(OPENROUTER_EMBEDDINGS_URL, json=payload, headers=headers)
                    resp.raise_for_status()

                data = resp.json()
                if "data" not in data:
                    error_msg = data.get("error", {}).get("message", str(data)) if isinstance(data, dict) else str(data)
                    raise RuntimeError(f"Embedding API error: {error_msg}")
                # Sort by index to guarantee order
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            except (httpx.HTTPStatusError, httpx.TimeoutException, RuntimeError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 64,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        vectors: list[list[float]] = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            batch_vectors = self._request(batch)
            vectors.extend(batch_vectors)
            if progress_callback:
                progress_callback(len(batch))
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(text_list):
                time.sleep(0.5)

        return vectors

    def embed_query(self, text: str) -> list[float]:
        result = self._request([text])
        return result[0]
