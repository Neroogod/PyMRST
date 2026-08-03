#!/usr/bin/env python3
"""Generate the MRST ad-mechanics to PyMRST parity ledger."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path


FIELDS = (
    "mrst_path", "symbol", "subsystem", "port_status", "python_counterpart",
    "validation_status", "validation_files", "notes",
)


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def executable_statements(path: Path) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )]


def is_alias_landing(path: Path) -> bool:
    statements = executable_statements(path)
    substantive = [node for node in statements if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return not substantive and all(isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) for node in statements)


def has_not_implemented(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, (ast.Call, ast.Name))
        and (
            isinstance(node.exc, ast.Name) and node.exc.id == "NotImplementedError"
            or isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "NotImplementedError"
        )
        for node in ast.walk(tree)
    )


def token(symbol: str, counterpart: Path | None = None) -> re.Pattern[str]:
    names = {symbol, snake_case(symbol)}
    if counterpart is not None:
        names.add(counterpart.stem)
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(map(re.escape, sorted(names))) + r")(?![A-Za-z0-9_])", re.I)


def generate(repo: Path, mrst_root: Path, python_root: Path) -> list[dict[str, str]]:
    python_by_stem: dict[str, list[Path]] = {}
    for path in python_root.rglob("*.py"):
        python_by_stem.setdefault(path.stem, []).append(path)
    validation_paths = sorted((repo / "tests").rglob("*.py")) + sorted((repo / "examples" / "validation").rglob("*.py"))
    validation_text = {p: p.read_text(encoding="utf-8", errors="ignore") for p in validation_paths if "parity" not in p.stem}
    rows: list[dict[str, str]] = []
    files = sorted(mrst_root.rglob("*.m")) + sorted(mrst_root.rglob("*.tex"))
    for source in sorted(files):
        rel = source.relative_to(mrst_root)
        symbol = source.stem
        counterpart = None
        status = "pending"
        note = "No matching Python owner"
        if source.suffix == ".tex":
            status, note = "excluded-asset", "Documentation asset"
        elif symbol == "Contents":
            status, note = "excluded-catalog", "MATLAB package catalog"
        else:
            matches = python_by_stem.get(snake_case(symbol), [])
            if not matches:
                matches = [
                    path for stem, paths in python_by_stem.items()
                    if normalized(stem) == normalized(symbol)
                    for path in paths
                ]
            if matches:
                # Prefer the counterpart in the equivalent relative directory.
                expected_parent = tuple(snake_case(x) for x in rel.parts[:-1])
                counterpart = next((p for p in matches if tuple(p.relative_to(python_root).parts[:-1]) == expected_parent), matches[0])
                if has_not_implemented(counterpart):
                    status = "stub"
                else:
                    status = "alias-landing" if is_alias_landing(counterpart) else "implemented-owner"
                note = "Named owner; behavioral validation tracked separately"
        pat = token(symbol, counterpart)
        validations = [p for p, text in validation_text.items() if pat.search(text)]
        validation_status = (
            "not-applicable" if status.startswith("excluded-")
            else "referenced" if validations else "unvalidated"
        )
        rows.append({
            "mrst_path": str(rel),
            "symbol": symbol,
            "subsystem": "/".join(rel.parts[:-1]) or "root",
            "port_status": status,
            "python_counterpart": str(counterpart.relative_to(repo)) if counterpart else "",
            "validation_status": validation_status,
            "validation_files": "; ".join(str(p.relative_to(repo)) for p in validations),
            "notes": note,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mrst", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    mrst = (args.mrst or repo.parent / "mrst-2024a-Geomech" / "modules" / "ad-mechanics").resolve()
    output = args.output or repo / "docs" / "ad_mechanics" / "parity.csv"
    rows = generate(repo, mrst, repo / "pymrst" / "modules" / "ad_mechanics")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["port_status"] for row in rows)
    validated = Counter(row["validation_status"] for row in rows)
    print(f"Wrote {len(rows)} rows to {output.relative_to(repo)}")
    print("Port status:", dict(sorted(counts.items())))
    print("Validation status:", dict(sorted(validated.items())))
    return 1 if counts["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
