from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@dataclass(slots=True)
class BookwormConfig:
    library_path: Path | None
    db_path: Path
    openrouter_api_key: str | None
    openrouter_model: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_env(cls) -> "BookwormConfig":
        load_dotenv()

        library = os.getenv("BOOKWORM_LIBRARY_PATH")
        db_path = os.getenv("BOOKWORM_DB_PATH", "./bookworm.duckdb")

        return cls(
            library_path=Path(library).expanduser().resolve() if library else None,
            db_path=Path(db_path).expanduser().resolve(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )

    def ensure_db_parent(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def require_library_path(self) -> Path:
        if not self.library_path:
            raise ValueError(
                "Library path is not configured. Set BOOKWORM_LIBRARY_PATH in .env or pass --path."
            )
        return self.library_path

    def require_openrouter(self) -> tuple[str, str]:
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        return self.openrouter_api_key, self.openrouter_model
