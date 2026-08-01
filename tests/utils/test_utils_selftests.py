"""Run every `pymrst/utils` module's own tests under pytest.

Each file in `pymrst/utils` carries its tests in an `if __name__ ==
'__main__'` block, which is the house convention and keeps a function's tests
next to the function.

**`tests/test_all.py` already does this for the whole of `pymrst/`**, and this
file deliberately overlaps it. Two reasons it still earns its place: that sweep
covers 946 modules and takes long enough that nobody runs it while working
(it does not finish inside ten minutes), whereas this covers the 56 under
`pymrst/utils` in about forty seconds; and it additionally checks each module
*reports* passing, which the repo-wide version does not.

Worth knowing when reading either: until the duplicate test basenames were
fixed, `pytest tests/` aborted during collection, so `tests/test_all.py` had
never actually run.

Each module runs as a **subprocess**, not an import. Its tests live under
`__main__`, so importing would not execute them; `runpy` would, but it would
also share this process's state, and several of these modules mutate grids and
emit warnings. A subprocess gives each one a clean interpreter and turns a
crash into a readable failure rather than taking the suite down.

The module list is discovered, not written out, so a new file is covered the
moment it lands. That is the point: the failure mode this guards against is
someone adding a module and forgetting to wire up its tests.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "pymrst" / "utils"


def _modules():
    """Every module under pymrst/utils that carries its own tests."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        if "__main__" not in path.read_text(errors="ignore"):
            continue
        found.append(path)
    return found


MODULES = _modules()


def test_the_discovery_found_something():
    """A typo in the glob would otherwise make this file pass vacuously."""
    assert len(MODULES) > 30, f"only found {len(MODULES)} modules under {PACKAGE}"


@pytest.mark.parametrize("module", MODULES,
                         ids=[str(p.relative_to(PACKAGE)) for p in MODULES])
def test_module_selftests(module):
    result = subprocess.run([sys.executable, str(module)], cwd=ROOT,
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-25:]
        pytest.fail(f"{module.relative_to(PACKAGE)} self-tests failed:\n"
                    + "\n".join(tail))


def test_every_module_reports_passing():
    """The self-tests print a final line; a module that silently does nothing
    would still exit 0, so check the output actually says it ran."""
    quiet = []
    for module in MODULES:
        result = subprocess.run([sys.executable, str(module)], cwd=ROOT,
                                capture_output=True, text=True, timeout=600)
        if "All tests passed" not in result.stdout:
            quiet.append(str(module.relative_to(PACKAGE)))
    assert not quiet, f"these ran but reported nothing: {quiet}"
