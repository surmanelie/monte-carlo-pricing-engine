"""Small argument checks that raise :class:`ValueError` with a clear message."""

from __future__ import annotations

import math
from collections.abc import Iterable


def require_finite(name: str, value: float) -> float:
    """Return ``float(value)`` or raise if it is not a finite real number."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def require_positive(name: str, value: float) -> float:
    """Return ``float(value)`` or raise if it is not finite and strictly positive."""
    x = require_finite(name, value)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return x


def require_non_negative(name: str, value: float) -> float:
    """Return ``float(value)`` or raise if it is negative or not finite."""
    x = require_finite(name, value)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return x


def require_int_at_least(name: str, value: int, minimum: int) -> int:
    """Return ``value`` or raise if it is not an integer ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_in_range(name: str, value: float, low: float, high: float) -> float:
    """Return ``float(value)`` or raise unless ``low <= value <= high``."""
    x = require_finite(name, value)
    if not low <= x <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}], got {value!r}")
    return x


def require_choice(name: str, value: str, choices: Iterable[str]) -> str:
    """Return ``value`` or raise if it is not one of ``choices``."""
    options = tuple(choices)
    if value not in options:
        raise ValueError(f"unsupported {name} {value!r}; expected one of {options}")
    return value
