"""MRST's `ThreePhaseBlackOilModel` protocol, ported.

`ThreePhaseBlackOilModelAD` inherited from `object`. It therefore reached none
of the shared AD-OO machinery and carried none of MRST's nine-method protocol -
the same structural finding as the two-phase model before its own migration,
and the same consequence: a model outside the hierarchy cannot use anything
written against it.

Inheriting `PhysicalModel` supplied five of the nine. The four here are the
black-oil-specific ones, and each is about a quantity the base class has no
notion of - dissolved gas and vaporised oil:

* `getDissolutionMatrix` - which component rides in which phase;
* `getDissolutionMatrixMax` - the same at saturated conditions, the ceiling a
  cell hits when it crosses the bubble point;
* `getScalingFactorsCPR` - the `1/b` weights that make the pressure system
  elliptic;
* `evalSingle` - MRST's one-region cell-array unwrapping.

The migration itself required fixing a probe first. `cfl.py` asked
`hasattr(model, "init_state_ad")` and then called it with one argument, while
`PhysicalModel.init_state_ad` takes three - so inheritance would have flipped
the probe `True` and the call would have raised. That is the third instance of
the same trap in this port, and `test_the_init_state_ad_probe_survives` pins
the fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModelAD,
)
from pymrst.modules.ad_core.models.physical_model import PhysicalModel


def _model(disgas=True, vapoil=False, phases=("W", "O", "G")):
    """A model with the protocol's inputs and nothing else built."""
    m = ThreePhaseBlackOilModelAD.__new__(ThreePhaseBlackOilModelAD)
    m.disgas, m.vapoil = disgas, vapoil
    names = list(phases)
    m.get_phase_names = lambda: names
    m.fluid = SimpleNamespace(
        rs_sat=lambda p: np.full(np.size(p), 50.0),
        rv_sat=lambda p: np.full(np.size(p), 2.0),
        b_w=lambda p: np.full(np.size(p), 1.05),
        b_o=lambda p, *a: np.full(np.size(p), 1.20),
        b_g=lambda p, *a: np.full(np.size(p), 0.90))
    m.raw_fluid = None
    return m


# -- the migration itself ----------------------------------------------------

def test_it_is_in_the_ad_oo_hierarchy():
    """The structural fix. Outside it, the class reaches none of the shared
    machinery - which is why nine methods were absent rather than five."""
    assert issubclass(ThreePhaseBlackOilModelAD, PhysicalModel)


def test_the_inherited_half_of_the_protocol_is_reachable():
    """These five come from `PhysicalModel` and needed no black-oil code."""
    for name in ("getPrimaryVariables", "getVariableField", "prepareTimestep",
                 "validateModel", "validateState"):
        assert hasattr(ThreePhaseBlackOilModelAD, name), name


def test_the_init_state_ad_probe_survives():
    """The blocker the migration had to clear first.

    `PhysicalModel.init_state_ad` takes three required arguments; `cfl.py`
    calls it with one. A `hasattr` probe would flip `True` on inheritance and
    the call would raise, so the probe must test whether the method is the
    *base's* rather than whether the name exists.
    """
    from pymrst.modules.ad_core.utils.cfl import _as_ad_state

    state = {"pressure": np.ones(3)}
    # A model that only inherits the base implementation must be left alone.
    inherited = ThreePhaseBlackOilModelAD.__new__(ThreePhaseBlackOilModelAD)
    assert _as_ad_state(inherited, state) is state

    # One with its own single-argument version must be used.
    class Own(PhysicalModel):
        def init_state_ad(self, state):
            return {"marked": True}

    assert _as_ad_state(Own.__new__(Own), state) == {"marked": True}


# -- dissolution matrix ------------------------------------------------------

def test_the_dissolution_matrix_places_rs_and_rv_in_their_own_slots():
    """`dismat[i][j]` is how much of component `j` rides in phase `i`. The
    conservation equations index it directly, so a misplaced entry adds
    solution gas to the wrong phase."""
    m = _model(disgas=True, vapoil=True)
    table = m.get_dissolution_matrix(rs=5.0, rv=0.5)

    names = m.get_phase_names()
    oil, gas = names.index("O"), names.index("G")
    assert table[oil][gas] == 0.5, "rv belongs in the oil-gas slot"
    assert table[gas][oil] == 5.0, "rs belongs in the gas-oil slot"
    assert table[names.index("W")][oil] is None


def test_a_dead_oil_model_drops_rs_however_it_is_called():
    """A model with no dissolved gas has none, whatever a caller passes.
    Keeping it would add solution gas to a fluid that has none."""
    table = _model(disgas=False, vapoil=True).get_dissolution_matrix(
        rs=5.0, rv=0.5)
    names = ["W", "O", "G"]
    assert table[names.index("G")][names.index("O")] is None
    assert table[names.index("O")][names.index("G")] == 0.5


def test_the_table_is_sized_by_the_active_phases():
    """One row per *active* phase. A three-phase table handed to a two-phase
    model would put the gas ratio in the oil slot."""
    table = _model(phases=("W", "O")).get_dissolution_matrix(rs=5.0)
    assert len(table) == 2 and all(len(row) == 2 for row in table)
    assert all(entry is None for row in table for entry in row)


def test_the_saturated_matrix_uses_the_bubble_point_ratios():
    """The ceiling a cell hits crossing the bubble point: above it, extra gas
    comes out as a free phase rather than staying dissolved."""
    m = _model(disgas=True, vapoil=True)
    table = m.get_dissolution_matrix_max(np.array([2.0e7]))

    names = m.get_phase_names()
    np.testing.assert_allclose(table[names.index("G")][names.index("O")], 50.0)
    np.testing.assert_allclose(table[names.index("O")][names.index("G")], 2.0)


# -- CPR scaling -------------------------------------------------------------

def _problem(pressure=2.0e7):
    return SimpleNamespace(state={"pressure": np.full(4, pressure)})


def test_the_cpr_weights_are_one_over_b():
    """The cancellation CPR depends on: equations are in surface volume
    `pv b s`, so dividing by `b` gives reservoir volume and the saturations
    sum to one. An unweighted sum cancels nothing."""
    m = _model()
    scaling = m.get_scaling_factors_cpr(
        _problem(), ["water", "oil", "gas"], "quasiimpes")

    assert scaling[0] == pytest.approx(1/1.05)
    assert scaling[1] == pytest.approx(1/1.20)
    assert scaling[2] == pytest.approx(1/0.90)


def test_an_unknown_equation_gets_unit_scaling():
    """A model may carry equations CPR knows nothing about - a well closure,
    a tracer. Unity leaves them alone rather than inventing a weight."""
    scaling = _model().get_scaling_factors_cpr(
        _problem(), ["water", "closureWells"], "quasiimpes")
    assert scaling[1] == 1


def test_a_live_oil_model_evaluates_b_at_the_saturated_ratio():
    """With dissolved gas the oil `b` depends on `rs`, so the weight has to be
    taken at the saturated ratio rather than the dead-oil curve."""
    m = _model(disgas=True)
    seen = {}

    def b_o(p, *args):
        seen["args"] = args
        return np.full(np.size(p), 1.20)

    m.fluid.b_o = b_o
    m.get_scaling_factors_cpr(_problem(), ["oil"], "quasiimpes")
    assert seen["args"], "b_o was called without rs"
    np.testing.assert_allclose(np.ravel(seen["args"][0])[0], 50.0)


def test_a_dead_oil_model_calls_b_without_a_ratio():
    m = _model(disgas=False)
    seen = {}

    def b_o(p, *args):
        seen["args"] = args
        return np.full(np.size(p), 1.20)

    m.fluid.b_o = b_o
    m.get_scaling_factors_cpr(_problem(), ["oil"], "quasiimpes")
    assert seen["args"] == ()


# -- evalSingle --------------------------------------------------------------

def test_eval_single_unwraps_a_one_region_list():
    """Regionalised properties arrive as lists even with a single region;
    without the unwrapping a list would be called as though it were a
    function."""
    assert ThreePhaseBlackOilModelAD.eval_single([lambda x: x*2], 3) == 6
    assert ThreePhaseBlackOilModelAD.eval_single(lambda x: x*2, 3) == 6


def test_the_mrst_spellings_all_resolve():
    """A caller coming from MATLAB uses these names."""
    for name in ("getDissolutionMatrix", "getDissolutionMatrixMax",
                 "getScalingFactorsCPR", "evalSingle"):
        assert hasattr(ThreePhaseBlackOilModelAD, name), name
