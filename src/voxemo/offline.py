"""Never contact the Hugging Face Hub once the models are on disk.

Even with a local copy, transformers checks the Hub for updates at every start
(and prints a warning about tokens). When every model is already cached, we set
``HF_HUB_OFFLINE=1`` before any Hugging Face library is imported: the assistant
then starts without network, as promised. The first start still downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

from voxemo.config import Settings


def hub_cache_dir() -> Path:
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    home = (
        os.environ.get("HF_HOME")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    )
    return Path(home) / "hub"


def _cached(repo_id: str, filename: str, cache: Path) -> bool:
    snapshots = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return any(snapshots.glob(f"*/{filename}"))


def required_models(settings: Settings) -> list[tuple[str, str]]:
    whisper = settings.whisper_model
    whisper_repo = whisper if "/" in whisper else f"Systran/faster-whisper-{whisper}"
    return [(whisper_repo, "model.bin"), (settings.emotion_model, "config.json")]


def prefer_local_models(settings: Settings) -> bool:
    """Switch the Hugging Face libraries to offline mode if everything is cached.

    Returns True when offline mode is on. Must run before transformers,
    faster-whisper or huggingface_hub are imported (they read it at import time).
    """
    if os.environ.get("HF_HUB_OFFLINE") not in (None, ""):
        return os.environ["HF_HUB_OFFLINE"] not in ("0", "false")
    cache = hub_cache_dir()
    if all(_cached(repo, name, cache) for repo, name in required_models(settings)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False
