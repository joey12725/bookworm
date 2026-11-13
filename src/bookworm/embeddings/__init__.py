"""Embedding providers and the factory that picks one from :class:`~bookworm.config.Settings`."""

from __future__ import annotations

from bookworm.config import PROVIDERS, Settings
from bookworm.embeddings.base import Embedder, l2_normalize
from bookworm.embeddings.fastembed import FastEmbedEmbedder
from bookworm.embeddings.hashing import HashingEmbedder
from bookworm.embeddings.openai_compat import OpenAICompatibleEmbedder
from bookworm.errors import BookwormError


def create_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder
    if kind == "hash":
        return HashingEmbedder(dim=settings.hash_dim)
    if kind == "fastembed":
        return FastEmbedEmbedder(model=settings.embed_model)
    if kind in PROVIDERS or kind == "openai-compatible":
        preset = PROVIDERS.get(kind, {})
        base_url = settings.embed_base_url or preset.get("base_url")
        model = settings.embed_model or preset.get("embed_model")
        if not base_url or not model:
            raise BookwormError(
                "The openai-compatible embedder needs BOOKWORM_EMBED_BASE_URL "
                "and BOOKWORM_EMBED_MODEL"
            )
        if preset.get("key_env") and not settings.embed_api_key:
            raise BookwormError(
                f"The {kind} embedder needs an API key: "
                f"set {preset['key_env']} or BOOKWORM_EMBED_API_KEY"
            )
        return OpenAICompatibleEmbedder(
            base_url=base_url, model=model, api_key=settings.embed_api_key
        )
    known = ", ".join(["hash", "fastembed", "openai-compatible", *PROVIDERS])
    raise BookwormError(f"Unknown embedder {kind!r}. Choose one of: {known}")


__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "OpenAICompatibleEmbedder",
    "create_embedder",
    "l2_normalize",
]
