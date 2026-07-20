"""Setup helper for MRST `setupSimpleCompositionalExample.m`."""

import copy
from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid
from pymrst.gridprocessing import compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.examples._shared import SimpleSchedule
from pymrst.modules.compositional.examples._shared import simple_schedule
from pymrst.modules.compositional.fluids import FluidCaseInfo
from pymrst.modules.compositional.model import CompositionalState
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import Rock
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import Well
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import DARCY
from pymrst.utils.units import DAY
from pymrst.utils.units import MILLI


@dataclass
class SimpleCompositionalSetup:
    """Grid, rock, model, wells, and initial state for the simple example."""

    grid: object
    rock: Rock
    mixture: Mixture
    info: FluidCaseInfo
    wells: list[Well]
    model: OverallCompositionModel
    state: CompositionalState
    schedule: SimpleSchedule
    reference: dict


def setup_simple_compositional_example(
    use_natural: bool = False,
    nx: int = 8,
    length: float = 100.0,
    well_rate: float = 5.0e-5,
    producer_bhp: float = 55.0e5,
    nstep: int = 1,
    timestep: float = 10.0 * DAY,
) -> SimpleCompositionalSetup:
    """Build MRST's `setupSimpleCompositionalExample` in native PyMRST form."""
    grid = compute_geometry(cart_grid([nx, 1, 1], [length, 10.0, 10.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    mixture = Mixture(
        names=("Methane", "CarbonDioxide", "n-Decane"),
        Tcrit=np.array([190.5640, 304.1282, 617.7000]),
        Pcrit=np.array([4599200.0, 7377300.0, 2103000.0]),
        Vcrit=np.array([9.8628e-05, 9.4118e-05, 6.0976e-04]),
        acentric=np.array([0.0114, 0.2239, 0.4884]),
        molarMass=np.array([0.0160, 0.0440, 0.1423]),
        name="MRST SIMPLE_COMP native mixture",
    )
    info = FluidCaseInfo(
        injection=np.array([0.0, 1.0, 0.0]),
        initial=np.array([0.6, 0.1, 0.3]),
        pressure=75.0e5,
        T=150.0 + 273.15,
    )

    wells = add_well(
        None,
        grid,
        rock,
        np.array([0]),
        well_type="rate",
        val=well_rate,
        compi=np.array([0.0, 1.0]),
        sign=1,
        name="Injector",
    )
    wells[0].components = info.injection
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([nx - 1]),
        well_type="bhp",
        val=producer_bhp,
        compi=np.array([0.0, 1.0]),
        sign=-1,
        name="Producer",
    )
    wells[1].components = info.initial
    for well in wells:
        well.lims = None

    model_cls = NaturalVariablesCompositionalModel if use_natural else OverallCompositionModel
    model = model_cls(
        grid,
        rock,
        mixture,
        temperature=info.T,
        wells=wells,
        rho_s=(800.0, 10.0),
    )
    state = model.initial_state(info.pressure, info.initial)
    schedule_wells = copy.deepcopy(wells)
    for well in schedule_wells:
        well.components = info.injection.copy()
        well.compi = np.array([1.0, 0.0])
        well.lims = None
    schedule = simple_schedule(np.full(int(nstep), float(timestep)), wells=schedule_wells)
    reference = {
        "available": False,
        "reason": "MRST simplecomp comparison.mat is not bundled with PyMRST",
        "mrst_file": "comparison.mat",
    }
    return SimpleCompositionalSetup(grid, rock, mixture, info, wells, model, state, schedule, reference)


def setupSimpleCompositionalExample(useNatural: bool = True, **kwargs):
    """MRST-style return signature for ``setupSimpleCompositionalExample``.

    MRST returns ``[state0, model, schedule, ref]``.  The Pythonic
    :func:`setup_simple_compositional_example` keeps the full setup object; this
    alias returns the MRST tuple so exact example ports can follow the MATLAB
    structure directly.
    """
    setup = setup_simple_compositional_example(use_natural=bool(useNatural), **kwargs)
    return setup.state, setup.model, setup.schedule, setup.reference


if __name__ == "__main__":
    def test_setup_has_injector_and_producer():
        setup = setup_simple_compositional_example(nx=6)
        assert setup.grid.cells.num == 6
        assert len(setup.wells) == 2
        assert setup.wells[0].sign == 1
        assert setup.wells[1].sign == -1
        assert setup.state.z.shape == (6, setup.mixture.n)
        assert np.allclose(setup.info.initial, [0.6, 0.1, 0.3])
        assert np.allclose(setup.info.injection, [0.0, 1.0, 0.0])
        assert setup.schedule.step.val.size == 1
        assert setup.schedule.control[0].wells is not setup.wells
        assert all(well.lims is None for well in setup.wells)
        assert np.allclose(setup.wells[1].components, setup.info.initial)
        assert all(np.allclose(well.components, [0.0, 1.0, 0.0])
                   for well in setup.schedule.control[0].wells)
        print("ok test_setup_has_injector_and_producer")

    def test_setup_can_build_natural_model():
        setup = setup_simple_compositional_example(True, nx=6)
        assert isinstance(setup.model, NaturalVariablesCompositionalModel)
        assert np.allclose(setup.wells[0].components, [0.0, 1.0, 0.0])
        print("ok test_setup_can_build_natural_model")

    def test_one_step_converges():
        setup = setup_simple_compositional_example(nx=6)
        state, ok, _ = setup.model.step(setup.state, 10.0 * 24.0 * 60.0 * 60.0)
        assert ok
        assert np.all(np.isfinite(state.pressure))
        assert np.allclose(state.z.sum(axis=1), 1.0)
        assert np.linalg.norm(state.z - setup.state.z) > 0.0
        print("ok test_one_step_converges")

    test_setup_has_injector_and_producer()
    test_setup_can_build_natural_model()
    test_one_step_converges()
    print("\nAll tests passed!")


__all__ = ["SimpleCompositionalSetup", "setupSimpleCompositionalExample", "setup_simple_compositional_example"]
