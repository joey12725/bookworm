from pathlib import Path

from bookworm.config import Settings, read_dotenv


def test_defaults_work_without_any_environment() -> None:
    s = Settings.from_env({})
    assert s.embedder == "hash"
    assert s.chat_provider is None
    assert s.db_path == Path("bookworm.duckdb")


def test_from_env_reads_overrides_and_provider_keys() -> None:
    s = Settings.from_env(
        {
            "BOOKWORM_DB": "/tmp/x.duckdb",
            "BOOKWORM_EMBEDDER": "openrouter",
            "BOOKWORM_CHAT_PROVIDER": "openai",
            "OPENROUTER_API_KEY": "or-test",
            "OPENAI_API_KEY": "oa-test",
            "BOOKWORM_CHUNK_TOKENS": "200",
            "BOOKWORM_EXCLUDE_DIRS": "comics, scans",
        }
    )
    assert s.db_path == Path("/tmp/x.duckdb")
    assert s.embed_api_key == "or-test"
    assert s.chat_api_key == "oa-test"
    assert s.chunk_tokens == 200
    assert s.excluded_dirs == ("comics", "scans")


def test_redacted_masks_keys() -> None:
    s = Settings(embed_api_key="secret-value")
    shown = s.redacted()
    assert shown["embed_api_key"] == "set"
    assert "secret-value" not in str(shown)


def test_read_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nexport BOOKWORM_DB="lib.duckdb"\nBOOKWORM_EMBEDDER=hash\nnoise\n')
    assert read_dotenv(env) == {"BOOKWORM_DB": "lib.duckdb", "BOOKWORM_EMBEDDER": "hash"}
    assert read_dotenv(tmp_path / "missing") == {}
