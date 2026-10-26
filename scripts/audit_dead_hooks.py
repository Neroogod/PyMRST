"""Find MRST-named methods that exist in this port but nothing ever calls.

`prepareReportstep` was defined, aliased, and covered by a protocol test that
asserted the name was present - while no code path invoked it. Every
per-report-step behaviour MRST performs was therefore skipped, and the one that
mattered (converting `resv_history` controls) sat behind four latent crashes
because it had never once executed.

A name-presence test cannot tell a live hook from a dead one. This can: it
reports methods whose only appearances are their own `def` and their
MRST-cased alias.

Run from the repo root:  python3 scripts/audit_dead_hooks.py
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where behaviour lives. Restricted to the solve loop rather than the whole
#: tree, because a helper nobody calls is untidy while a *hook* nobody calls is
#: silently-skipped physics.
SCOPE = [
    "pymrst/modules/ad_core/models",
    "pymrst/modules/ad_core/simulators",
    "pymrst/modules/ad_core/solvers",
]

#: Searched for call sites: the library plus its tests and examples, so a
#: method exercised only by a test still counts as reachable.
SEARCH = ["pymrst", "tests", "examples", "scripts"]

#: Dunders, and names Python itself or a framework calls for us.
IGNORE = re.compile(r"^(_|test_)")
FRAMEWORK = {"setUp", "tearDown", "main"}


def python_files(rels):
    for rel in rels:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def defined_methods():
    """{name: [(file, line, class)]} for every method defined in SCOPE."""
    out = defaultdict(list)
    for path in python_files(SCOPE):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if IGNORE.match(item.name) or item.name in FRAMEWORK:
                        continue
                    out[item.name].append(
                        (path.relative_to(ROOT), item.lineno, node.name))
    return out


def reference_counts(names):
    """How many times each name appears as something other than its own
    definition or a bare alias assignment."""
    counts = defaultdict(int)
    alias = {n: re.compile(rf"^\s*\w+\s*=\s*{re.escape(n)}\s*$") for n in names}
    define = {n: re.compile(rf"^\s*(async\s+)?def\s+{re.escape(n)}\b")
              for n in names}
    word = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    for path in python_files(SEARCH):
        for line in path.read_text(errors="ignore").splitlines():
            for name in names:
                if not word[name].search(line):
                    continue
                if define[name].match(line) or alias[name].match(line):
                    continue
                counts[name] += 1
    return counts


def main() -> int:
    methods = defined_methods()
    counts = reference_counts(list(methods))
    dead = sorted(n for n in methods if counts[n] == 0)

    print(f"Scanned {len(methods)} public methods across {len(SCOPE)} package "
          f"trees.\n")
    if not dead:
        print("No unreachable methods found.")
        return 0

    print(f"UNREACHABLE - defined, never referenced anywhere ({len(dead)}):\n")
    for name in dead:
        for rel, line, cls in methods[name]:
            print(f"  {cls}.{name}\n      {rel}:{line}")
    print("\nEach is either dead code or a hook whose caller was never ported."
          "\nCheck against MRST: if MRST calls its counterpart, the behaviour "
          "is being skipped.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
