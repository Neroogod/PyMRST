"""MRST ``runNorneExample.m`` compatibility wrapper."""

import importlib.util
from pathlib import Path


def _load_script():
    here = Path(__file__).resolve()
    script = here.parents[4] / "examples" / "run_norne_geomech.py"
    spec = importlib.util.spec_from_file_location("pymrst_run_norne_geomech",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_norne_example(norne_case: str = "mini Norne", **kwargs):
    """Run PyMRST's Norne geomechanics example with MRST-shaped naming."""
    mod = _load_script()
    subset = "full" if str(norne_case).lower() == "full" else "mini"
    return mod.run(subset=subset, **kwargs)


__all__ = ["run_norne_example"]


if __name__ == "__main__":
    run_norne_example(n_steps=1)
