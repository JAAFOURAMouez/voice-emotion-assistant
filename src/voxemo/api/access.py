"""Optional access code (``ACCESS_TOKEN``) for a server reachable from outside.

REST calls send it as ``Authorization: Bearer <code>``. Browsers cannot add headers
to a WebSocket, so the page offers it as a subprotocol: ``["voxemo", "token.<code>"]``;
the server answers with the plain ``voxemo`` subprotocol, never echoing the code.
"""

from __future__ import annotations

import hmac

SUBPROTOCOL = "voxemo"
_TOKEN_PREFIX = "token."
WS_ACCESS_DENIED = 4401
"""WebSocket close code: wrong or missing access code (the page asks for it)."""


def token_matches(given: str | None, expected: str) -> bool:
    """Constant-time comparison; always true when no code is configured."""
    if not expected:
        return True
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


def bearer(header: str | None) -> str | None:
    if header and header[:7].lower() == "bearer ":
        return header[7:].strip()
    return None


def subprotocols(header: str | None) -> tuple[bool, str | None]:
    """(``voxemo`` offered, code offered) from ``Sec-WebSocket-Protocol``."""
    offered = [p.strip() for p in (header or "").split(",") if p.strip()]
    code = next((p[len(_TOKEN_PREFIX) :] for p in offered if p.startswith(_TOKEN_PREFIX)), None)
    return SUBPROTOCOL in offered, code
