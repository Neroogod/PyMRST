"""Shared guards for MRST examples whose data/import substrate is absent."""


def unsupported_example(name: str, reason: str):
    raise NotImplementedError(
        f"{name} is present in the PyMRST layout, but cannot run yet: {reason}"
    )


__all__ = ["unsupported_example"]
