from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    convert_incomp_well_sols,
    get_perforation_to_well_mapping,
    get_report_ministeps,
)
from pymrst.params.wells_and_bc import Well


def _well(name, cells, compi):
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells, type="rate", val=1.0, r=np.full(cells.size, 0.1),
        dir="z", WI=np.ones(cells.size), dZ=np.zeros(cells.size), name=name,
        compi=np.asarray(compi, float), refDepth=0.0, sign=1,
    )


def test_perforation_mapping_and_sparse_incidence():
    wells = [_well("A", [0, 1], [1, 0]), _well("B", [2], [0, 1])]
    mapping, incidence = get_perforation_to_well_mapping(wells)
    assert np.array_equal(mapping, [0, 0, 1])
    assert sp.issparse(incidence)
    assert np.array_equal(incidence.toarray(), [[1, 0], [1, 0], [0, 1]])


def test_incompressible_conversion_uses_fractional_flow_for_production():
    wells = [_well("A", [0, 1], [1, 0]), _well("B", [2], [0.25, 0.75])]
    state = SimpleNamespace(wellSol=[
        SimpleNamespace(flux=np.array([-2.0, 3.0]), pressure=100.0),
        SimpleNamespace(flux=np.array([-4.0]), pressure=90.0),
    ])

    class Fluid:
        def properties(self, state):
            return np.array([1.0, 2.0])

        def saturation(self, state):
            return np.zeros((3, 2))

        def relperm(self, saturation, state):
            return np.array([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])

    rows = convert_incomp_well_sols(wells, [state], Fluid())[0]
    # Cell 0 fractional flow = [2/3, 1/3], plus water-only injection.
    assert np.allclose([rows[0].qWs, rows[0].qOs], [5 / 3, -2 / 3])
    # Cell 2 fractional flow = [0.75, 0.25].
    assert np.allclose([rows[1].qWs, rows[1].qOs], [-3.0, -1.0])
    assert rows[1].bhp == 90.0


def test_report_ministeps_only_returns_converged_attempts():
    report = SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=False, Timestep=4.0),
            SimpleNamespace(Converged=True, Timestep=2.0),
        ]),
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=True, Timestep=3.0),
        ]),
    ])
    assert np.array_equal(get_report_ministeps(report), [2.0, 3.0])
