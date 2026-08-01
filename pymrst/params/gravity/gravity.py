"""Stateful compatibility implementation of MRST ``gravity.m``."""

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from pymrst.params.gravity_const import NORM_GRAVITY, gravity_or_zero, gravity_vector

_strength = NORM_GRAVITY
_direction = np.array([0.0, 0.0, 1.0])
_enabled = False


def gravity(*controls) -> np.ndarray:
    """Update or query process-local MRST-compatible gravity state."""
    global _strength, _direction, _enabled
    for control in controls:
        if isinstance(control, str):
            mode = control.lower()
            if mode in ("off", "no", "false"):
                _enabled = False
            elif mode in ("on", "yes", "true"):
                _enabled = True
            elif mode in ("x", "y", "z"):
                _direction = np.zeros(3)
                _direction["xyz".index(mode)] = 1.0
            elif mode == "reset":
                _reset()
            else:
                raise ValueError(f"unsupported gravity control mode {control!r}")
            continue
        if isinstance(control, (bool, np.bool_)):
            _enabled = bool(control)
            continue
        values = np.asarray(control)
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("gravity control must be a string, boolean, or real number")
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 1:
            norm = float(np.linalg.norm(_direction))
            if norm == 0.0:
                raise ValueError("gravity direction is the zero vector; reset it first")
            _strength = float(values[0])
            _direction = _direction/norm
        elif values.size in (2, 3):
            _strength = 1.0
            _direction = np.pad(values, (0, 3-values.size))
        else:
            raise ValueError("gravity vector must contain two or three components")
    return current_gravity()


def current_gravity(griddim: int = 3) -> np.ndarray:
    """Return a copy of the active vector, truncated to ``griddim``."""
    if griddim not in (1, 2, 3):
        raise ValueError("griddim must be 1, 2, or 3")
    vector = _strength*_direction if _enabled else np.zeros(3)
    return vector[:griddim].copy()


def _reset() -> None:
    global _strength, _direction, _enabled
    _strength = NORM_GRAVITY
    _direction = np.array([0.0, 0.0, 1.0])
    _enabled = False


@contextmanager
def isolated_gravity(*controls) -> Iterator[np.ndarray]:
    """Temporarily modify gravity state and restore it on context exit."""
    global _strength, _direction, _enabled
    saved = (_strength, _direction.copy(), _enabled)
    try:
        yield gravity(*controls)
    finally:
        _strength, _direction, _enabled = saved


__all__ = [
    "NORM_GRAVITY", "current_gravity", "gravity", "gravity_or_zero",
    "gravity_vector", "isolated_gravity",
]
