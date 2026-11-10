"""Tests for application settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from voxemo.config import Settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ignore the developer's real .env and environment."""
    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_defaults_are_local_and_private() -> None:
    settings = Settings()
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2"
    assert settings.language == "fr"
    assert settings.store_raw_audio is False
    assert settings.emotion_confidence_threshold == 0.5
    assert settings.whisper_compute_type == "int8"


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("STORE_RAW_AUDIO", "true")
    monkeypatch.setenv("LANGUAGE", " EN ")
    settings = Settings()
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.store_raw_audio is True
    assert settings.language == "en"


def test_reads_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WHISPER_MODEL=base\nDB_PATH=x/y.db\n", encoding="utf-8")
    settings = Settings()
    assert settings.whisper_model == "base"
    assert settings.db_path == Path("x/y.db")


def test_env_example_matches_settings() -> None:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    keys = {
        line.split("=", 1)[0].lower()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert keys == set(Settings.model_fields)


@pytest.mark.parametrize("value", ["-0.1", "1.5"])
def test_rejects_out_of_range_threshold(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("EMOTION_CONFIDENCE_THRESHOLD", value)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_remote_ollama_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "https://llm.example.com")
    with pytest.raises(ValidationError, match="not this machine"):
        Settings()


def test_remote_ollama_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.20:11434/")
    monkeypatch.setenv("ALLOW_REMOTE_OLLAMA", "true")
    assert Settings().ollama_host == "http://192.168.1.20:11434"


def test_rejects_malformed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    with pytest.raises(ValidationError):
        Settings()
