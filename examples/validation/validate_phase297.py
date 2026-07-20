"""Phase 297: MRST GenericOverallCompositionModel layout."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.components import ImmiscibleComponent  # noqa: E402
from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example  # noqa: E402
from pymrst.modules.compositional.models import GenericOverallCompositionModel  # noqa: E402
from pymrst.modules.compositional.models.contents import list_models  # noqa: E402
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def build_generic(water: bool = False):
    setup = setup_simple_compositional_example(use_natural=False, nx=5, nstep=1)
    model = GenericOverallCompositionModel(
        setup.grid,
        setup.rock,
        setup.mixture,
        temperature=setup.info.T,
        wells=() if water else setup.wells,
        rho_s=(800.0, 10.0),
        water=water,
    )
    state = model.initial_state(setup.info.pressure, setup.info.initial, sW=0.15 if water else None)
    return setup, model, state


def validate_generic_surface() -> int:
    _, model, state = build_generic(water=False)
    variables, names, origin = model.getPrimaryVariables(state)
    failures = 0
    failures += _check("OutputStateFunctions match MRST generic overall defaults",
                       model.OutputStateFunctions == ["ComponentTotalMass", "Density"])
    failures += _check("phase names and indices match oil/gas generic overall layout",
                       model.getPhaseNames() == ("O", "G")
                       and model.getLiquidIndex() == 0
                       and model.getVaporIndex() == 1)
    failures += _check("component names default to EOS component names",
                       model.getComponentNames() == tuple(model.mix.names))
    failures += _check("primary variables are pressure plus all but first component",
                       names == ["pressure", *list(model.mix.names[1:])]
                       and len(variables) == len(names)
                       and origin == ["GenericOverallCompositionModel"] * len(names))
    return failures


def validate_component_setup_and_state_ad() -> int:
    _, model, state = build_generic(water=True)
    model.validateModel()
    variables, names, origin = model.getPrimaryVariables(state)
    out = model.initStateAD(state, variables, names, origin)
    failures = 0
    failures += _check("validateModel creates water plus EOS components",
                       isinstance(model.Components[0], ImmiscibleComponent)
                       and all(isinstance(c, EquationOfStateComponent) for c in model.Components[1:])
                       and model.getComponentNames()[0] == "water")
    failures += _check("water model exposes W/O/G phase names",
                       model.getPhaseNames() == ("W", "O", "G")
                       and model.getNonEoSPhaseNames() == ("W",))
    failures += _check("primary variables include water saturation when active",
                       names[-1] == "sW"
                       and np.allclose(variables[-1], state.sW))
    failures += _check("initStateAD reconstructs normalized composition",
                       np.allclose(out.z.sum(axis=1), 1.0)
                       and np.all(out.z >= 0.0)
                       and np.allclose(out.sW, state.sW))
    return failures


def validate_cleanup_and_metadata() -> int:
    _, model, state = build_generic(water=False)
    state.FractionalDerivatives = np.ones((state.pressure.size, 2))
    updated, report = model.updateState(state)
    state.dz = np.ones_like(state.z)
    state.dpRel = np.ones_like(state.pressure)
    converged, report2 = model.updateAfterConvergence(None, state, 1.0, None)
    meta = list_models()["GenericOverallCompositionModel"]
    failures = 0
    failures += _check("updateState removes FractionalDerivatives",
                       updated is state and report is None and not hasattr(state, "FractionalDerivatives"))
    failures += _check("updateAfterConvergence removes temporary nonlinear fields",
                       converged is state and report2 is None
                       and not hasattr(state, "dz")
                       and not hasattr(state, "dpRel"))
    failures += _check("contents metadata points at generic_overall_composition_model",
                       meta["python_module"] == "generic_overall_composition_model"
                       and meta["status"] == "implemented_generic_surface")
    return failures


def main() -> int:
    print("\nPhase 297: MRST GenericOverallCompositionModel layout")
    failures = validate_generic_surface()
    failures += validate_component_setup_and_state_ad()
    failures += validate_cleanup_and_metadata()
    if failures:
        print(f"\nPhase 297 failed ({failures} checks).")
        return 1
    print("\nGenericOverallCompositionModel now has the MRST generic surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
