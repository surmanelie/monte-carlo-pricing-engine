from __future__ import annotations

import pytest

from mcengine._validation import (
    require_choice,
    require_in_range,
    require_int_at_least,
    require_non_negative,
    require_positive,
)


def test_helpers_accept_valid_values() -> None:
    assert require_positive("x", 1) == 1.0
    assert require_non_negative("x", 0.0) == 0.0
    assert require_int_at_least("n", 3, 2) == 3
    assert require_in_range("rho", -0.5, -1.0, 1.0) == -0.5
    assert require_choice("kind", "a", ("a", "b")) == "a"


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (require_positive, ("x", float("inf"))),
        (require_non_negative, ("x", -1e-9)),
        (require_int_at_least, ("n", 2.0, 1)),
        (require_int_at_least, ("n", True, 0)),
        (require_int_at_least, ("n", 1, 2)),
        (require_in_range, ("rho", 1.5, -1.0, 1.0)),
        (require_choice, ("kind", "c", ("a", "b"))),
    ],
)
def test_helpers_reject_invalid_values(func: object, args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match=r"must|unsupported"):
        func(*args)  # type: ignore[operator]
