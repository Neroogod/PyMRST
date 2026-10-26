"""Read-only deck inspection, include lineage, and keyword lookup tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


_KEYWORD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]{0,7})(?:\s|$)")
_SECTIONS = frozenset(
    {"RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE"}
)


class DeckAccessError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise DeckAccessError(f"deck path lies outside configured roots: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class DeckOccurrence:
    keyword: str
    section: str
    source: str
    line: int
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword, "section": self.section,
            "source": self.source, "line": self.line, "checksum": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class DeckIndex:
    deck_id: str
    root_path: str
    root_checksum: str
    files: tuple[tuple[str, str], ...]
    occurrences: tuple[DeckOccurrence, ...]


class DeckCatalog:
    """Security boundary and immutable source index for text decks."""

    def __init__(
        self,
        allowed_roots: Iterable[str | Path],
        *,
        maximum_files: int = 512,
        maximum_total_bytes: int = 128 * 1024**2,
        maximum_include_depth: int = 64,
    ) -> None:
        if isinstance(allowed_roots, (str, Path)):
            allowed_roots = (allowed_roots,)
        self.roots = tuple(Path(item).resolve() for item in allowed_roots)
        if not self.roots:
            raise ValueError("deck catalog requires at least one allowed root")
        self.maximum_files = int(maximum_files)
        self.maximum_total_bytes = int(maximum_total_bytes)
        self.maximum_include_depth = int(maximum_include_depth)
        if min(self.maximum_files, self.maximum_total_bytes, self.maximum_include_depth) <= 0:
            raise ValueError("deck catalog limits must be positive")
        self._decks: dict[str, DeckIndex] = {}

    def register(self, deck_id: str, path: str | Path) -> DeckIndex:
        deck_id = str(deck_id).strip()
        if not deck_id or deck_id in self._decks:
            raise KeyError(f"deck id is empty or already registered: {deck_id}")
        root = _within(Path(path), self.roots)
        if not root.is_file():
            raise FileNotFoundError(root)
        state = {"bytes": 0, "files": [], "occurrences": []}
        self._scan(root, state, stack=(), section="", depth=0)
        index = DeckIndex(
            deck_id, str(root), _sha256(root), tuple(state["files"]),
            tuple(state["occurrences"]),
        )
        self._decks[deck_id] = index
        return index

    def get(self, deck_id: str) -> DeckIndex:
        try:
            return self._decks[deck_id]
        except KeyError as error:
            raise KeyError(f"unknown deck: {deck_id}") from error

    def _scan(self, path, state, *, stack, section, depth):
        path = _within(path, self.roots)
        if path in stack:
            chain = " -> ".join(str(item) for item in stack + (path,))
            raise DeckAccessError(f"INCLUDE cycle: {chain}")
        if depth > self.maximum_include_depth:
            raise DeckAccessError("maximum INCLUDE depth exceeded")
        size = path.stat().st_size
        state["bytes"] += size
        if state["bytes"] > self.maximum_total_bytes:
            raise DeckAccessError("deck and INCLUDE files exceed byte limit")
        if len(state["files"]) >= self.maximum_files:
            raise DeckAccessError("deck exceeds INCLUDE file-count limit")
        checksum = _sha256(path)
        state["files"].append((str(path), checksum))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        index = 0
        current_section = section
        while index < len(lines):
            cleaned = lines[index].split("--", 1)[0].strip()
            match = _KEYWORD.match(cleaned) if cleaned and not cleaned.startswith(("'", '"')) else None
            if not match:
                index += 1
                continue
            keyword = match.group(1).upper()
            if keyword in _SECTIONS:
                current_section = keyword
            state["occurrences"].append(
                DeckOccurrence(keyword, current_section, str(path), index + 1, checksum)
            )
            if keyword != "INCLUDE":
                index += 1
                continue
            include_line = index + 1
            while include_line < len(lines):
                record = lines[include_line].split("--", 1)[0].strip()
                if record:
                    break
                include_line += 1
            if include_line >= len(lines):
                raise DeckAccessError(f"unterminated INCLUDE at {path}:{index + 1}")
            name = record.rstrip("/").strip().strip("'\"").strip()
            if not name:
                raise DeckAccessError(f"empty INCLUDE at {path}:{index + 1}")
            included = _within(path.parent / name, self.roots)
            if not included.is_file():
                raise FileNotFoundError(f"INCLUDE file not found: {included}")
            current_section = self._scan(
                included, state, stack=stack + (path,), section=current_section,
                depth=depth + 1,
            )
            index = include_line + 1
        return current_section

    def summary(self, deck_id: str) -> dict[str, Any]:
        deck = self.get(deck_id)
        counts = Counter(item.keyword for item in deck.occurrences)
        sections = sorted({item.section for item in deck.occurrences if item.section})
        return {
            "deck_id": deck.deck_id, "root_path": deck.root_path,
            "root_checksum": deck.root_checksum,
            "file_count": len(deck.files), "files": [
                {"path": path, "checksum": checksum} for path, checksum in deck.files
            ],
            "sections": sections, "keyword_counts": dict(sorted(counts.items())),
            "occurrence_count": len(deck.occurrences),
            "evidence": {
                "kind": "input", "uri": f"deck://{deck.deck_id}",
                "label": f"Deck {deck.deck_id}", "locator": {},
                "checksum": deck.root_checksum,
            },
        }

    def keyword(self, deck_id: str, keyword: str, *, maximum_occurrences=100):
        deck = self.get(deck_id)
        keyword = str(keyword).strip().upper()
        if not keyword:
            raise ValueError("keyword cannot be empty")
        matches = [item for item in deck.occurrences if item.keyword == keyword]
        if len(matches) > int(maximum_occurrences):
            raise DeckAccessError(
                f"keyword {keyword} has {len(matches)} occurrences; "
                f"limit is {maximum_occurrences}"
            )
        return {
            "deck_id": deck_id, "keyword": keyword, "count": len(matches),
            "occurrences": [item.to_dict() for item in matches],
            "evidence": [
                {
                    "kind": "deck_record",
                    "uri": f"deck://{deck_id}/{keyword}",
                    "label": f"{keyword} at {Path(item.source).name}:{item.line}",
                    "locator": {
                        "source": item.source, "line": item.line,
                        "section": item.section, "keyword": keyword,
                    },
                    "checksum": item.checksum,
                }
                for item in matches
            ],
        }


class InspectDeckTool:
    descriptor = ToolDescriptor(
        "inspect_deck", "Inspect deck sections, keyword counts, and include lineage",
        AgentPermission.INSPECT,
    )

    def __init__(self, catalog: DeckCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(call.call_id, True, self.catalog.summary(str(call.arguments["deck_id"])))


class GetKeywordTool:
    descriptor = ToolDescriptor(
        "get_keyword", "Locate a keyword in registered deck sources",
        AgentPermission.INSPECT,
    )

    def __init__(self, catalog: DeckCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        value = self.catalog.keyword(
            str(call.arguments["deck_id"]), str(call.arguments["keyword"]),
            maximum_occurrences=int(call.arguments.get("maximum_occurrences", 100)),
        )
        return ToolResult(call.call_id, True, value)


__all__ = [
    "DeckAccessError", "DeckCatalog", "DeckIndex", "DeckOccurrence",
    "GetKeywordTool", "InspectDeckTool",
]
