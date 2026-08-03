"""MRST ``runNorneExample.m`` compatibility wrapper."""

import importlib.util
from pathlib import Path


class NorneBranchUnavailable(RuntimeError):
    """Requested Norne physics branch is not connected to the runner."""


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
    fluid_model = str(kwargs.pop("fluid_model", "oil water")).lower()
    method = str(kwargs.pop("method", "fully coupled")).lower()
    bc_case = str(kwargs.pop("bc_case", "bottom fixed")).lower()
    splitting_tolerance = float(kwargs.pop("splitting_tolerance", 1.0e-6))
    if fluid_model not in {"water", "oil water", "oil-water", "oil_water", "blackoil", "black oil"}:
        raise NorneBranchUnavailable(
            f"Norne fluid_model={fluid_model!r} is not connected to the "
            "field runner yet"
        )
    fully_coupled = method in {"fully coupled", "fully_coupled"}
    fixed_stress = method in {"fixed stress", "fixed stress splitting", "fixed_stress_splitting"}
    split_fluid = fluid_model in {"water", "oil water", "oil-water", "oil_water", "blackoil", "black oil"}
    if not fully_coupled and not (split_fluid and fixed_stress):
        raise NorneBranchUnavailable(
            f"Norne method={method!r} is not connected to the field runner yet"
        )
    if bc_case != "bottom fixed":
        raise NorneBranchUnavailable(
            f"Norne bc_case={bc_case!r} is not connected to the field runner yet"
        )
    return mod.run(
        subset=subset, fluid_model=fluid_model, method=method,
        splitting_tolerance=splitting_tolerance, **kwargs,
    )


__all__ = ["NorneBranchUnavailable", "run_norne_example"]


if __name__ == "__main__":
    run_norne_example(n_steps=1)
