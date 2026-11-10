"""Offline mode once the models are cached."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxemo.config import Settings
from voxemo.offline import prefer_local_models, required_models


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return tmp_path


def _fill(cache: Path, repo: str, filename: str) -> None:
    folder = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    (folder / filename).write_text("x")


def test_offline_when_every_model_is_cached(cache: Path) -> None:
    import os

    settings = Settings(_env_file=None)
    for repo, filename in required_models(settings):
        _fill(cache, repo, filename)
    assert prefer_local_models(settings) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_stays_online_for_the_first_download(cache: Path) -> None:
    import os

    settings = Settings(_env_file=None)
    _fill(cache, "Systran/faster-whisper-small", "model.bin")  # emotion model missing
    assert prefer_local_models(settings) is False
    assert "HF_HUB_OFFLINE" not in os.environ


def test_explicit_choice_is_respected(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert prefer_local_models(Settings(_env_file=None)) is False


def test_whisper_repo_names() -> None:
    assert required_models(Settings(_env_file=None, whisper_model="base"))[0] == (
        "Systran/faster-whisper-base",
        "model.bin",
    )
    custom = Settings(_env_file=None, whisper_model="someone/whisper-fr")
    assert required_models(custom)[0][0] == "someone/whisper-fr"
