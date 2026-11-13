"""Settings, read from environment variables (and an optional ``.env`` file).

Every setting has a working default, so ``bookworm index ./books`` runs with zero setup on
the built-in offline embedder. Point the embedder or the chat model at any
OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio, vLLM...) to upgrade.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

#: Known OpenAI-compatible providers: base URL, API-key env var, default embed/chat models.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4o-mini",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "embed_model": "openai/text-embedding-3-small",
        "chat_model": "openai/gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "embed_model": "nomic-embed-text",
        "chat_model": "llama3.2",
    },
}


@dataclass(slots=True)
class Settings:
    """All runtime configuration. Construct directly in code, or via :meth:`from_env`."""

    db_path: Path = field(default_factory=lambda: Path("bookworm.duckdb"))
    library_path: Path | None = None

    # Embeddings: "hash" (built-in, offline), "fastembed" (local ONNX), or an
    # OpenAI-compatible provider name from PROVIDERS ("openai", "openrouter", "ollama").
    embedder: str = "hash"
    embed_model: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    hash_dim: int = 512

    # Grounded Q&A. With no chat provider, `ask` answers extractively from the passages.
    chat_provider: str | None = None
    chat_model: str | None = None
    chat_base_url: str | None = None
    chat_api_key: str | None = None

    # Chunking
    chunk_tokens: int = 320
    chunk_overlap: int = 64

    # Retrieval
    rrf_k: int = 60
    #: 1.0 disables MMR; around 0.7 trades some relevance for more varied results.
    mmr_lambda: float = 1.0

    excluded_dirs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Accept plain strings from Python callers.
        self.db_path = Path(self.db_path)
        if self.library_path is not None:
            self.library_path = Path(self.library_path)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, dotenv: Path | None = Path(".env")
    ) -> Settings:
        values: dict[str, str] = {}
        if dotenv is not None and env is None:
            values.update(read_dotenv(dotenv))
        values.update(os.environ if env is None else env)

        def get(name: str) -> str | None:
            value = values.get(name)
            return value if value not in (None, "") else None

        kwargs: dict[str, Any] = {}
        if db := get("BOOKWORM_DB"):
            kwargs["db_path"] = Path(db).expanduser()
        if lib := get("BOOKWORM_LIBRARY"):
            kwargs["library_path"] = Path(lib).expanduser()
        for name, env_name in {
            "embedder": "BOOKWORM_EMBEDDER",
            "embed_model": "BOOKWORM_EMBED_MODEL",
            "embed_base_url": "BOOKWORM_EMBED_BASE_URL",
            "embed_api_key": "BOOKWORM_EMBED_API_KEY",
            "chat_provider": "BOOKWORM_CHAT_PROVIDER",
            "chat_model": "BOOKWORM_CHAT_MODEL",
            "chat_base_url": "BOOKWORM_CHAT_BASE_URL",
            "chat_api_key": "BOOKWORM_CHAT_API_KEY",
        }.items():
            if (value := get(env_name)) is not None:
                kwargs[name] = value
        for name, env_name in {
            "hash_dim": "BOOKWORM_HASH_DIM",
            "chunk_tokens": "BOOKWORM_CHUNK_TOKENS",
            "chunk_overlap": "BOOKWORM_CHUNK_OVERLAP",
        }.items():
            if (value := get(env_name)) is not None:
                kwargs[name] = int(value)
        if (mmr := get("BOOKWORM_MMR_LAMBDA")) is not None:
            kwargs["mmr_lambda"] = float(mmr)
        if excluded := get("BOOKWORM_EXCLUDE_DIRS"):
            kwargs["excluded_dirs"] = tuple(d.strip() for d in excluded.split(",") if d.strip())

        settings = cls(**kwargs)
        # Fall back to the provider's conventional key variable (OPENAI_API_KEY etc.).
        if settings.embed_api_key is None:
            settings.embed_api_key = _provider_key(settings.embedder, values)
        if settings.chat_provider and settings.chat_api_key is None:
            settings.chat_api_key = _provider_key(settings.chat_provider, values)
        return settings

    def with_overrides(self, **changes: Any) -> Settings:
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    def redacted(self) -> dict[str, str]:
        """Settings as display strings, with secrets masked."""
        out: dict[str, str] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("api_key"):
                out[f.name] = "set" if value else "-"
            elif isinstance(value, tuple):
                out[f.name] = ", ".join(value) or "-"
            else:
                out[f.name] = "-" if value is None else str(value)
        return out


def _provider_key(provider: str, values: Mapping[str, str]) -> str | None:
    key_env = PROVIDERS.get(provider, {}).get("key_env")
    return values.get(key_env) or None if key_env else None


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Enough for a settings file; no interpolation."""
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result
