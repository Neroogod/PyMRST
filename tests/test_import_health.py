"""Guard against import cycles and layering inversions across `pymrst`.

This exists because of a concrete bug. `expandMatrixToCell` was implemented in
`modules/compositional/utils/` with `modules/ad_core/` re-exporting it - a
lower-level package importing a higher-level one. That closed a cycle, so
`import pymrst.modules.compositional.model` raised ImportError, and **273 tests
could not run**. The defect sat there undetected because the only thing that
would have caught it was a suite that could be collected, and duplicate test
basenames were preventing that.

Two guards, both cheap:

- every module that takes part in an import cycle must still import from a
  **cold interpreter**. That is the exact check the compositional bug failed;
  a cycle only bites when some import order reaches it first, so importing each
  participant first is what surfaces it.
- the number of cycles must not grow. Cycles here are mostly the benign
  "package `__init__` imports its submodule, submodule imports back from the
  package" shape, which works until it does not. Ratcheting the count means a
  new one has to be argued for rather than merely not noticed.

Neither guard forbids the existing cycles: unpicking six vemmech modules is a
real refactor and not this test's business. The point is that the next one gets
noticed on the day it lands.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
from collections import defaultdict

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pymrst"

# Cycles present when this guard was written. Lowering it is progress; raising
# it means a new layering inversion arrived.
KNOWN_CYCLES = 5


def _module_name(path: pathlib.Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_graph():
    """Module-level `pymrst` imports only - deferred ones cannot cycle."""
    graph = defaultdict(set)
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        me = _module_name(path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("pymrst"):
                    graph[me].add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("pymrst"):
                        graph[me].add(alias.name)
    return graph


def _cycles(graph):
    """Strongly connected components of size > 1, iteratively (the graph is
    deep enough that a recursive Tarjan can exhaust the stack)."""
    index, low, on_stack, stack, found, counter = {}, {}, set(), [], [], [0]
    for root in list(graph):
        if root in index:
            continue
        work = [(root, iter(graph.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter[0]
                    counter[0] += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(graph.get(child, ()))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    top = stack.pop()
                    on_stack.discard(top)
                    component.append(top)
                    if top == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
    return found


GRAPH = _import_graph()
CYCLES = _cycles(GRAPH)
IN_A_CYCLE = sorted({m for cycle in CYCLES for m in cycle})


def test_the_graph_was_actually_built():
    """A broken parse would make every other test here pass vacuously."""
    assert len(GRAPH) > 100, f"only {len(GRAPH)} modules have pymrst imports"


@pytest.mark.parametrize("module", IN_A_CYCLE)
def test_a_cycle_member_imports_from_cold(module):
    """The check the compositional bug failed."""
    result = subprocess.run([sys.executable, "-c", f"import {module}"],
                            cwd=ROOT, capture_output=True, text=True,
                            timeout=300)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-3:]
        pytest.fail(f"`import {module}` fails from a cold interpreter:\n"
                    + "\n".join(tail))


def test_no_new_import_cycles():
    summary = "\n".join(
        "  " + " -> ".join(c[:4]) + (" ..." if len(c) > 4 else "")
        for c in sorted(CYCLES, key=len, reverse=True))
    assert len(CYCLES) <= KNOWN_CYCLES, (
        f"{len(CYCLES)} import cycles, up from {KNOWN_CYCLES}. A new one means "
        f"a module imports something that imports it back - usually a "
        f"lower-level package reaching up. Cycles now:\n{summary}")


def test_utils_never_imports_a_module():
    """`pymrst/utils` is the bottom of the stack and must stay there.

    `pymrst/utils/inout` had to move `read_keywords` out of
    `modules/deckformat` for exactly this reason: the upward import was not
    merely untidy, it deadlocked the import.
    """
    upward = sorted(
        f"{src} -> {dst}"
        for src, targets in GRAPH.items() if src.startswith("pymrst.utils")
        for dst in targets if dst.startswith("pymrst.modules"))
    assert not upward, (
        "pymrst.utils imports from pymrst.modules at module level:\n  "
        + "\n  ".join(upward))
