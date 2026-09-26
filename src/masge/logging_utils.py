"""Credential-shaped token redaction for persisted error messages."""

import re

_API_KEY_PATTERN = re.compile(r"(?:sk|rk|key)-[A-Za-z0-9_-]{8,}")


def redact_text(value: str) -> str:
    """Remove common credential-shaped tokens before logging persisted text."""
    return _API_KEY_PATTERN.sub("[REDACTED]", value)
