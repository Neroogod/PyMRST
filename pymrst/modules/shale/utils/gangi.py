"""Gangi pressure-dependent permeability multiplier."""


def gangi_fn(p, pc, alpha, pmax, m, k0, matrix_perm):
    """Evaluate MRST shale ``gangiFn`` without stripping AD information."""
    if pmax == 0:
        raise ValueError("pmax must be non-zero")
    return (k0 / matrix_perm) * (1 - ((pc - alpha * p) / pmax) ** m) ** 3


__all__ = ["gangi_fn"]
