"""Phase 112: K-value EOS backend inside native compositional flow equations.

Phase 111 validates the converted SPE1 K-value/property tables against MRST.
This phase exercises the next layer: those tables are passed to PyMRST's native
overall-composition conservation equations through ``eos_model=...``.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial
from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.fluids.blackoil import convert_black_oil_model_to_compositional_model
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI


def _kvalue_model(model_cls=OverallCompositionModel):
    _, blackoil_model, _ = setup_spe1_tutorial()
    comp_model = convert_black_oil_model_to_compositional_model(
        blackoil_model,
        np_=20,
        nz=20,
        interpolation="linear",
    )
    grid = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    rho_s = (blackoil_model.fluid.rho_s[1], blackoil_model.fluid.rho_s[2])
    model = model_cls(
        grid,
        rock,
        comp_model.mixture,
        temperature=273.15,
        rho_s=rho_s,
        eos_model=comp_model.eos_model,
        max_iterations=20,
        tol_cnv=1.0e-7,
    )
    return model


def _total_component_mass(model, state):
    fp = backend_flow_props(model.mix, model.eos_model, state.pressure, model.T, state.z)
    return np.array([
        np.sum(model.pv * (fp.rhoL * fp.sL * fp.xM[i] + fp.rhoV * fp.sV * fp.yM[i]))
        for i in range(model.mix.n)
    ])


def main() -> None:
    model = _kvalue_model(OverallCompositionModel)
    state0 = model.initial_state(np.linspace(220.0, 200.0, 3) * BARSA, [0.5, 0.5])
    mass0 = _total_component_mass(model, state0)
    state1, ok, iterations = model.step(state0, 1.0 * DAY)
    if not ok:
        raise AssertionError("K-value overall-composition step did not converge")
    mass1 = _total_component_mass(model, state1)
    if not np.allclose(mass1, mass0, rtol=2.0e-6, atol=1.0e-8):
        raise AssertionError((mass0, mass1, mass1 - mass0))
    if not np.allclose(state1.z.sum(axis=1), 1.0):
        raise AssertionError("overall compositions drifted off the simplex")

    natural = _kvalue_model(NaturalVariablesCompositionalModel)
    nstate0 = natural.initial_state(np.linspace(220.0, 200.0, 3) * BARSA, [0.5, 0.5])
    nstate1, nok, natural_iterations = natural.step(nstate0, 0.25 * DAY)
    if not nok:
        raise AssertionError("K-value natural-variable step did not converge")
    if not np.allclose(nstate1.z.sum(axis=1), 1.0):
        raise AssertionError("natural overall compositions drifted off the simplex")
    if not np.allclose(nstate1.sO + nstate1.sG, 1.0):
        raise AssertionError("natural hydrocarbon saturations drifted off closure")

    print(
        "phase112 K-value flow backend passed "
        f"(overall {iterations} Newton iterations, natural {natural_iterations})"
    )


if __name__ == "__main__":
    main()
