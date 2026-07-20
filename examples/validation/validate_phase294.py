"""Phase 294: MRST-layout ThreePhaseCompositionalModel base surface."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.models import ThreePhaseCompositionalModel  # noqa: E402
from pymrst.modules.compositional.models.contents import list_models  # noqa: E402
from pymrst.modules.compositional.models.eos import EquilibriumConstantModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ConstantPropertyModel:
    def computeMolarDensity(self, eos, p, x, Z, T, isLiquid):
        del eos, x, Z, T
        return np.full(np.asarray(p, dtype=float).reshape(-1).shape, 800.0 if isLiquid else 20.0)


def build_model(water: bool = True) -> ThreePhaseCompositionalModel:
    mix = simple_mixture()
    eos = EquilibriumConstantModel(
        None,
        mix,
        [
            lambda P, T, z: np.full_like(P, 0.5, dtype=float),
            lambda P, T, z: np.full_like(P, 1.2, dtype=float),
            lambda P, T, z: np.full_like(P, 2.0, dtype=float),
        ],
    )
    eos.PropertyModel = ConstantPropertyModel()
    grid = SimpleNamespace(cells=SimpleNamespace(num=2))
    return ThreePhaseCompositionalModel(grid, None, None, eos, water=water)


def validate_names_and_indices() -> int:
    model = build_model(water=True)
    failures = 0
    failures += _check("model exports MRST phase labels",
                       model.getPhaseNames() == ("W", "O", "G"))
    failures += _check("liquid and vapour indices are zero-based O/G positions",
                       model.getLiquidIndex() == 1 and model.getVaporIndex() == 2)
    failures += _check("non-EOS phase names exclude oil and gas",
                       model.getNonEoSPhaseNames() == ("W",))
    failures += _check("EOS component mask includes water false then EOS true",
                       np.array_equal(model.getEoSComponentMask(), np.array([False, True, True, True])))
    failures += _check("component names include non-EOS water plus EOS names",
                       model.getComponentNames()[0] == "water"
                       and model.getComponentNames()[1:] == tuple(simple_mixture().names))
    return failures


def validate_variable_fields() -> int:
    model = build_model(water=True)
    failures = 0
    failures += _check("components alias maps to state.components",
                       model.getVariableField("z")[0] == "components")
    failures += _check("liquid/vapour mole fraction aliases map to x/y",
                       model.getVariableField("liquidmf")[0] == "x"
                       and model.getVariableField("vapormf")[0] == "y")
    failures += _check("saturation aliases map to liquid/vapour phase columns",
                       model.getVariableField("sl") == ("s", 1)
                       and model.getVariableField("sv") == ("s", 2))
    failures += _check("EOS component name maps into hydrocarbon composition columns",
                       model.getVariableField("n-Butane") == ("components", 1))
    failures += _check("equilibrium constants alias maps to K",
                       model.getVariableField("equilibriumConstants")[0] == "K")
    return failures


def validate_state_flash_and_cleanup() -> int:
    model = build_model(water=False)
    state = {
        "pressure": np.array([1.0e7, 1.1e7]),
        "T": 300.0,
        "components": np.array([[0.35, 0.25, 0.40], [0.20, 0.35, 0.45]]),
        "dz": np.ones((2, 3)),
        "dpRel": np.ones(2),
    }
    validated = model.validateState(state)
    updated, report = model.updateAfterConvergence(None, validated, 1.0, None)
    failures = 0
    failures += _check("validateState expands scalar temperature",
                       validated["T"].shape == (2,) and np.allclose(validated["T"], 300.0))
    failures += _check("validateState flashes missing x/y/K fields",
                       all(key in validated for key in ("x", "y", "K", "L", "Z_L", "Z_V"))
                       and np.allclose(validated["x"].sum(axis=1), 1.0)
                       and np.allclose(validated["y"].sum(axis=1), 1.0))
    failures += _check("updateAfterConvergence removes MRST temporary fields",
                       updated is validated and report is None
                       and "dz" not in updated
                       and "dpRel" not in updated)
    return failures


def validate_contents_metadata() -> int:
    meta = list_models()["ThreePhaseCompositionalModel"]
    return _check("contents metadata points at the real base-surface module",
                  meta["python_module"] == "three_phase_compositional_model"
                  and meta["status"] == "implemented_base_surface")


def main() -> int:
    print("\nPhase 294: MRST-layout ThreePhaseCompositionalModel base surface")
    failures = validate_names_and_indices()
    failures += validate_variable_fields()
    failures += validate_state_flash_and_cleanup()
    failures += validate_contents_metadata()
    if failures:
        print(f"\nPhase 294 failed ({failures} checks).")
        return 1
    print("\nThreePhaseCompositionalModel base compatibility surface is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
