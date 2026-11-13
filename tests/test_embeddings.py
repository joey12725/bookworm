from __future__ import annotations

import json
import math
import sys
import types
from typing import Any

import httpx
import pytest

from bookworm.config import Settings
from bookworm.embeddings import (
    FastEmbedEmbedder,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)
from bookworm.errors import BookwormError, ProviderError


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=128)
    a = emb.embed_query("The lighthouse keeper")
    assert a == HashingEmbedder(dim=128).embed_query("The lighthouse keeper")
    assert len(a) == 128
    assert math.isclose(math.sqrt(dot(a, a)), 1.0, rel_tol=1e-9)
    assert emb.name == "hash:v1:128"


def test_hashing_embedder_ranks_related_text_higher() -> None:
    emb = HashingEmbedder()
    query = emb.embed_query("marrying daughters")
    related, unrelated = emb.embed_documents(
        ["She hoped one daughter would marry him.", "The army crossed the river at dawn."]
    )
    assert dot(query, related) > dot(query, unrelated)


def test_hashing_embedder_empty_text() -> None:
    assert set(HashingEmbedder(dim=16).embed_query("the of and")) == {0.0}


def _embedding_handler(calls: list[dict[str, Any]], fail_first: int = 0):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append({"body": body, "auth": request.headers.get("authorization")})
        if len(calls) <= fail_first:
            return httpx.Response(429, text="slow down")
        data = [
            {"index": i, "embedding": [float(len(text)), 0.0, 0.0]}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return handler


def test_openai_compatible_batches_orders_and_normalises() -> None:
    calls: list[dict[str, Any]] = []
    client = httpx.Client(transport=httpx.MockTransport(_embedding_handler(calls)))
    emb = OpenAICompatibleEmbedder(
        base_url="https://api.example.test/v1/", model="m", api_key="k", batch_size=2, client=client
    )
    vectors = emb.embed_documents(["a", "bb", "ccc"])
    assert len(calls) == 2
    assert calls[0]["auth"] == "Bearer k"
    assert calls[0]["body"] == {"model": "m", "input": ["a", "bb"]}
    assert vectors == [[1.0, 0.0, 0.0]] * 3
    assert emb.name == "openai-compatible:m"


def test_openai_compatible_retries_rate_limits() -> None:
    calls: list[dict[str, Any]] = []
    client = httpx.Client(transport=httpx.MockTransport(_embedding_handler(calls, fail_first=2)))
    emb = OpenAICompatibleEmbedder(base_url="https://x.test", model="m", client=client, backoff=0)
    assert emb.embed_query("hi") == [1.0, 0.0, 0.0]
    assert len(calls) == 3


def test_openai_compatible_does_not_retry_auth_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OpenAICompatibleEmbedder(base_url="https://x.test", model="m", client=client, backoff=0)
    with pytest.raises(ProviderError, match="401"):
        emb.embed_query("hi")
    assert calls == 1


def test_fastembed_adapter_with_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class Vec(list[float]):
        def tolist(self) -> list[float]:
            return list(self)

    class TextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str], batch_size: int = 32):
            return (Vec([3.0, 4.0]) for _ in texts)

        def query_embed(self, texts: list[str]):
            return iter([Vec([0.0, 2.0])])

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding)
    )
    emb = FastEmbedEmbedder()
    assert emb.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert emb.embed_query("q") == [0.0, 1.0]
    assert emb.name == "fastembed:BAAI/bge-small-en-v1.5"


def test_create_embedder() -> None:
    assert isinstance(create_embedder(Settings()), HashingEmbedder)
    ollama = create_embedder(Settings(embedder="ollama"))
    assert isinstance(ollama, OpenAICompatibleEmbedder)
    assert ollama.base_url == "http://localhost:11434/v1"
    with pytest.raises(BookwormError, match="OPENAI_API_KEY"):
        create_embedder(Settings(embedder="openai"))
    with pytest.raises(BookwormError, match="Unknown embedder"):
        create_embedder(Settings(embedder="nope"))
