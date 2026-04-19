from __future__ import annotations

from urllib.parse import urlparse

_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})


class NetworkGuardError(Exception):
    pass


def assert_localhost(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise NetworkGuardError(f"Network access to '{host}' blocked — only localhost allowed.")
