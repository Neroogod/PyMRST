"""Select the first automatic-differentiation value from candidate inputs."""

from __future__ import annotations


def get_sample_ad(*values):
    """Return ``(sample, is_ad)`` using the first AD-like candidate."""
    if not values:
        raise ValueError("at least one candidate is required")
    for candidate in values:
        if hasattr(candidate, "val") and hasattr(candidate, "jac"):
            return candidate, True
    return values[0], False


getSampleAD = get_sample_ad

__all__ = ["get_sample_ad", "getSampleAD"]
