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


def _module_scripts():
    skip = {"__init__.py"}
    return sorted(
        p for p in (ROOT / "pymrst").rglob("*.py")
        if p.name not in skip and "__pycache__" not in p.parts
    )


def _validators():
    return sorted(VALID.glob("validate_phase*.py"))


@pytest.mark.parametrize("script", _module_scripts(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_module_selftests(script):
    """Every module's own unit tests must pass."""
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
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
