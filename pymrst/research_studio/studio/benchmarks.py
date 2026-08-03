"""M5.4 versioned evidence portal with explicit publication state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from pymrst.research_studio.evidence.manifest import sha256_file


@dataclass(frozen=True, slots=True)
class BenchmarkEntry:
    benchmark_id: str
    title: str
    version: str
    evidence_path: str
    evidence_sha256: str
    publication_status: str = "internal"
    summary: str = ""

    def __post_init__(self):
        if self.publication_status not in {"internal", "candidate", "public", "withdrawn"}:
            raise ValueError("invalid benchmark publication status")

    @classmethod
    def from_path(cls, benchmark_id, title, version, path, *,
                  publication_status="internal", summary=""):
        source = Path(path).resolve()
        json.loads(source.read_text(encoding="utf-8"))
        return cls(benchmark_id, title, version, str(source), sha256_file(source),
                   publication_status, summary)

    def to_dict(self, *, include_path=True):
        value = {name: getattr(self, name) for name in self.__slots__}
        if not include_path:
            value.pop("evidence_path")
        return value


class BenchmarkPortal:
    def __init__(self, entries=()):
        self._entries = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry):
        key = (entry.benchmark_id, entry.version)
        if key in self._entries:
            raise KeyError(f"benchmark version already registered: {key}")
        self._entries[key] = entry

    def entries(self, *, public_only=False):
        values = tuple(self._entries[key] for key in sorted(self._entries))
        return tuple(value for value in values
                     if not public_only or value.publication_status == "public")

    def evidence(self, benchmark_id, version, *, public_request=False):
        entry = self._entries[(benchmark_id, version)]
        if public_request and entry.publication_status != "public":
            raise PermissionError("benchmark evidence is not public")
        source = Path(entry.evidence_path)
        if sha256_file(source) != entry.evidence_sha256:
            raise ValueError("benchmark evidence changed after registration")
        return json.loads(source.read_text(encoding="utf-8"))

    def promote(self, benchmark_id, version, status, *, explicit_approval=False):
        from dataclasses import replace
        key = (benchmark_id, version)
        current = self._entries[key]
        if status == "public" and not explicit_approval:
            raise PermissionError("public promotion requires explicit approval")
        updated = replace(current, publication_status=status)
        self._entries[key] = updated
        return updated


__all__ = ["BenchmarkEntry", "BenchmarkPortal"]
