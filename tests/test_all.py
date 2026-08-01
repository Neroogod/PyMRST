"""Run every PyMRST module's self-tests, and every MATLAB validation.

    pytest                    # module self-tests (fast, no MATLAB needed)
    pytest -m validation      # cross-checks against the saved MATLAB data
    pytest -q                 # everything

Each module under `pymrst/` is executable and runs its own unit tests in a
`__main__` block; this collects them so a single command checks the whole
project. The validation tests replay the saved MATLAB reference data in
`examples/validation/` - they do not need MATLAB itself, only the .mat
files produced earlier by the dump_phase*.m scripts.
"""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "examples" / "validation"


def _all_scripts():
    skip = {"__init__.py"}
    return sorted(
        p for p in (ROOT / "pymrst").rglob("*.py")
        if p.name not in skip and "__pycache__" not in p.parts
    )


def _is_example(path) -> bool:
    """Example drivers run a whole simulation; module tests do not."""
    return "examples" in path.parts


def _module_scripts():
    return [p for p in _all_scripts() if not _is_example(p)]


def _example_scripts():
    return [p for p in _all_scripts() if _is_example(p)]


def _validators():
    return sorted(VALID.glob("validate_phase*.py"))


@pytest.mark.parametrize("script", _module_scripts(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_module_selftests(script):
    """Every module's own unit tests must pass.

    A few modules are command-line tools rather than test suites - their
    `__main__` builds an ArgumentParser and requires an argument. Running one
    bare makes argparse print usage and exit 2, which is the tool working
    correctly, not a failing test. That exact signal is skipped rather than
    skipping every module that mentions argparse: two of the three that do have
    defaults and run their tests fine, so a blanket exclusion would lose real
    coverage.
    """
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    if r.returncode == 2 and "the following arguments are required" in r.stderr:
        pytest.skip(f"{script.name} is a command-line tool, not a self-test")
    assert r.returncode == 0, (
        f"{script.relative_to(ROOT)} failed:\n{r.stdout[-3000:]}\n"
        f"{r.stderr[-3000:]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("script", _example_scripts(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_example_drivers(script):
    """Every example driver must still run end to end.

    These are separated from the module self-tests for one reason: they run
    real simulations and some take far longer than a unit test should. SPE9
    completes all 35 of its steps but needs well over the 900s the module sweep
    allows, so it was failing on the clock rather than on anything real.

    They are **not** deselected by default - a failure here is a failure, and
    hiding it behind an opt-in marker is how the last batch of them went
    unnoticed. The `slow` marker exists so a developer can say
    `-m "not slow"` deliberately, not so CI can skip them by accident.
    """
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode == 2 and "the following arguments are required" in r.stderr:
        pytest.skip(f"{script.name} is a command-line tool, not a self-test")
    assert r.returncode == 0, (
        f"{script.relative_to(ROOT)} failed:\n{r.stdout[-3000:]}\n"
        f"{r.stderr[-3000:]}"
    )


@pytest.mark.validation
@pytest.mark.parametrize("script", _validators(),
                         ids=lambda p: p.stem.replace("validate_", ""))
def test_matlab_validation(script):
    """Every phase must still match its saved MATLAB reference."""
    ref_missing = not any(VALID.glob("ref_*.mat"))
    if ref_missing:
        pytest.skip("no MATLAB reference data present")
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    if "missing (run" in r.stdout:
        pytest.skip(f"reference data for {script.stem} not generated")
    assert r.returncode == 0, (
        f"{script.name} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    )
