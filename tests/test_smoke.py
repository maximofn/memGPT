from memgpt import __version__
from memgpt.config import Settings, get_settings


def test_version_is_set():
    assert __version__ == "0.1.0"


def test_settings_defaults_load():
    s = Settings(_env_file=None)
    assert s.primary_llm_model
    assert s.summarizer_llm_model
    assert s.postgres_dsn.startswith("postgresql://")
    assert ":5433/" in s.postgres_dsn
    assert s.neo4j_uri.startswith("bolt://")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
