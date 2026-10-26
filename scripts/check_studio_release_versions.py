#!/usr/bin/env python3
"""Fail a release when Python, npm, Cargo, and Tauri versions diverge."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib


def versions(root: Path) -> dict[str, str]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    python_source = ast.parse((root / "pymrst" / "__init__.py").read_text())
    python_version = ""
    for statement in python_source.body:
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in statement.targets)):
            python_version = ast.literal_eval(statement.value)
            break
    return {
        "pyproject": str(pyproject["project"]["version"]),
        "python": python_version,
        "npm": str(json.loads((root / "ui" / "package.json").read_text())["version"]),
        "cargo": str(tomllib.loads((root / "ui" / "src-tauri" / "Cargo.toml").read_text())
                     ["package"]["version"]),
        "tauri": str(json.loads((root / "ui" / "src-tauri" / "tauri.conf.json").read_text())
                     ["version"]),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = versions(root)
    if not all(found.values()) or len(set(found.values())) != 1:
        raise SystemExit("release version mismatch: " + json.dumps(found, sort_keys=True))
    print(json.dumps({"status": "consistent", "version": next(iter(found.values())),
                      "sources": found}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
