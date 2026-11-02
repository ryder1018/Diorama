"""Tuning knobs read from environment variables, falling back to the defaults in code."""

import os


def env_number(name: str, default: int | float) -> int | float:
    """$name parsed as the default's type (int or float); the default if unset or not a number."""
    raw = os.getenv(name, "").strip()
    try:
        return type(default)(raw) if raw else default
    except ValueError:
        return default


def env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """$name as a comma-separated list; the default if unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())
