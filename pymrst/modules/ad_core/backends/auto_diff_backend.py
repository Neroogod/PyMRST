"""Base automatic-differentiation backend descriptor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoDiffBackend:
    """MRST-style AD backend configuration.

    The descriptor records backend capabilities. Numerical assembly is performed
    by PyMRST's native sparse ADI implementation unless `available` is false, in
    which case benchmark examples report the backend as unsupported.
    """

    name: str = "AutoDiffBackend"
    use_mex: bool = False
    row_major: bool = False
    deferred_assembly: bool = False
    available: bool = True
    reason: str = ""

    def label(self) -> str:
        """Human-readable benchmark label."""
        return self.name

    def with_options(self, **kwargs) -> "AutoDiffBackend":
        """Return a copy with updated options."""
        data = {
            "name": self.name,
            "use_mex": self.use_mex,
            "row_major": self.row_major,
            "deferred_assembly": self.deferred_assembly,
            "available": self.available,
            "reason": self.reason,
        }
        data.update(kwargs)
        return type(self)(**data)


__all__ = ["AutoDiffBackend"]
