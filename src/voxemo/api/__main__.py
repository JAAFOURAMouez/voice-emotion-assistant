"""Start the web server: ``python -m voxemo.api`` then open http://localhost:8000."""

from __future__ import annotations

import uvicorn

from voxemo.api.app import create_app
from voxemo.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
