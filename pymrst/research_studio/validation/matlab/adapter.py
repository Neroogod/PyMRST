"""High-level MATLAB/MRST fixture adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .fixtures import generate_fixture, validate_fixture
from .schemas import MatlabFixtureManifest, MatlabFixtureSpec


class MatlabReferenceAdapter:
    def __init__(self, spec: MatlabFixtureSpec,
                 loader: Callable[[Path], Any]):
        self.spec = spec
        self.loader = loader

    def validate(self) -> MatlabFixtureManifest:
        manifest = validate_fixture(self.spec)
        assert manifest is not None
        return manifest

    def regenerate(self, *, executable: Path | str | None = None,
                   timeout_seconds: float = 1800.0) -> MatlabFixtureManifest:
        return generate_fixture(
            self.spec, executable=executable, timeout_seconds=timeout_seconds)

    def load(self) -> tuple[Any, MatlabFixtureManifest]:
        manifest = self.validate()
        return self.loader(self.spec.reference_path), manifest


__all__ = ["MatlabReferenceAdapter"]
