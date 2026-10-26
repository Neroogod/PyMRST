"""Equilibration regions (MRST ad-core/initialization).

Ports MRST's `getInitializationRegionsBase.m` / `getInitializationRegionsBlackOil.m`,
`getMinMaxPhaseSaturations.m`, and the multi-region path of `initStateBlackOilAD`
(AC16): the *region* layer on top of PyMRST's single-region equilibration
(Phase 42, `init_state_equil`).

A field is rarely one equilibrium region. Different fault blocks have different
fluid contacts and datum pressures; ECLIPSE's `EQLNUM` assigns each cell to a
region, and each region equilibrates independently on its own cells. This module
builds the region structs (with MRST's defaults) and assembles a full-field
initial state from a list of them - each region equilibrated on its cell subset,
scattered back into the field arrays.

`get_min_max_phase_saturations_ad` returns the per-phase saturation endpoints
(connate water; the mobile range), used to clamp the equilibrium saturations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.initialization.equilibrium import (
    EquilibriumRegion,
    init_state_equil,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.units import ATM


def get_min_max_phase_saturations_ad(model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Per-phase ``(s_min, s_max)`` endpoints (MRST `getMinMaxPhaseSaturations`).

    Water carries its connate saturation as a floor; the other phases run from
    zero to the complement of the connate water. This is the unscaled default;
    endpoint scaling (`rock.krscale`) is not present in PyMRST.
    """
    nph = (int(model.n_phases) if hasattr(model, "n_phases")
           else int(model.getNumberOfPhases()))
    raw = getattr(model, "raw_fluid", model.fluid)
    swcon = float(getattr(raw, "swcon", 0.0) or 0.0)
    s_min = np.zeros(nph)
    s_max = np.ones(nph)
    s_min[0] = swcon                       # connate water
    return s_min, s_max


def get_initialization_regions_base_ad(
    model: Any,
    rho: Sequence[Any],
    contacts: Sequence[float],
    datum_depth: float = 0.0,
    datum_pressure: float = ATM,
    cells: Optional[np.ndarray] = None,
    pc_functions: Optional[List[Any]] = None,
    pc_sign: Optional[np.ndarray] = None,
    contacts_pc: Optional[np.ndarray] = None,
    reference_index: Optional[int] = None,
    pc_scale: Optional[np.ndarray] = None,
    s_min: Optional[np.ndarray] = None,
    s_max: Optional[np.ndarray] = None,
) -> EquilibriumRegion:
    """Build one equilibration region with MRST's defaults.

    ``contacts`` are the ``n_phases - 1`` phase-contact depths. ``rho`` is the
    per-phase density (a value or callable per phase, as
    `initialize_equilibrium_pressures_ad` expects).
    """
    nph = (int(model.n_phases) if hasattr(model, "n_phases")
           else int(model.getNumberOfPhases()))
    contacts = np.asarray(contacts, dtype=float).ravel()
    assert contacts.size == nph - 1, \
        f"need {nph - 1} contacts for {nph} phases, got {contacts.size}"
    if s_min is None or s_max is None:
        default_min, default_max = get_min_max_phase_saturations_ad(model)
        s_min = default_min if s_min is None else s_min
        s_max = default_max if s_max is None else s_max
    reg = EquilibriumRegion(
        datum_depth=datum_depth,
        datum_pressure=datum_pressure,
        contacts=contacts,
        cells=(None if cells is None else np.asarray(cells, np.int64).ravel()),
        pc_functions=pc_functions,
        pc_sign=(np.ones(nph) if pc_sign is None else np.asarray(pc_sign, float)),
        contacts_pc=(np.zeros(nph - 1) if contacts_pc is None
                     else np.asarray(contacts_pc, float)),
        s_min=s_min,
        s_max=s_max,
        reference_index=reference_index,
        pc_scale=pc_scale,
    )
    reg.rho = list(rho)                    # carried for the pressure integration
    return reg


def init_state_equil_regions(model: Any,
                             regions: Sequence[EquilibriumRegion],
                             gravity: float) -> State:
    """Equilibrate a field made of several regions and assemble the full state.

    Each region is equilibrated on its own ``cells`` (which must partition the
    grid); the per-region results are scattered into the field-length arrays.
    """
    nc = model.G.cells.num
    covered = np.zeros(nc, dtype=bool)
    out = State(pressure=np.zeros(nc), sW=np.zeros(nc))
    has3 = model.n_phases > 2
    if has3:
        out["sO"] = np.zeros(nc)
        out["sG"] = np.zeros(nc)
    if model.fluid.disgas:
        out["rs"] = np.zeros(nc)
        out["status"] = np.ones(nc, dtype=np.int64)
    if model.fluid.vapoil:
        # `init_state_equil` computes `rv` for a wet-gas model; without a slot
        # here it was computed per region and then dropped on the way out, so
        # a `vapoil` model came back with no vaporised oil at all. Norne is
        # one, and the omission surfaced only in CPR's weighting, which reads
        # `state["rv"]` directly.
        out["rv"] = np.zeros(nc)
        out.setdefault("status", np.ones(nc, dtype=np.int64))

    for reg in regions:
        cells = (np.arange(nc) if reg.cells is None
                 else np.asarray(reg.cells, np.int64).ravel())
        sub = init_state_equil(model, reg, gravity)
        out["pressure"][cells] = sub["pressure"]
        out["sW"][cells] = sub["sW"]
        if has3:
            out["sO"][cells] = sub["sO"]
            out["sG"][cells] = sub["sG"]
        if model.fluid.disgas:
            out["rs"][cells] = sub["rs"]
            out["status"][cells] = sub["status"]
        if model.fluid.vapoil:
            out["rv"][cells] = sub["rv"]
            if "status" in sub:
                out["status"][cells] = sub["status"]
        covered[cells] = True

    assert covered.all(), "the regions must cover every cell of the grid"
    return out


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.

__all__ = [
    "get_initialization_regions_base_ad",
    "get_min_max_phase_saturations_ad",
    "init_state_equil_regions",
]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry, tensor_grid
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, POISE

    G_ACC = 9.80665

    def _model(nz=20, top=1000.0, thick=100.0):
        z = np.linspace(top, top + thick, nz + 1)
        g = compute_geometry(tensor_grid(np.array([0.0, 100.0]),
                                         np.array([0.0, 100.0]), z))
        rock = make_rock(g, 100 * MILLI_DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
        return g, GenericBlackOilModelAD(g, rock, fluid, [], gravity=G_ACC)

    MILLI_DARCY = 1e-3 * DARCY

    def _rho(model):
        return [1000.0, 800.0]              # constant phase densities

    def test_factory_reproduces_hand_built_region():
        """The factory region equilibrates identically to a hand-built one."""
        g, model = _model()
        woc = 1050.0
        reg = get_initialization_regions_base_ad(model, _rho(model), [woc],
                                              datum_depth=1000.0,
                                              datum_pressure=200 * BARSA)
        hand = EquilibriumRegion(datum_depth=1000.0, datum_pressure=200 * BARSA,
                                 contacts=[woc])
        hand.rho = _rho(model)
        a = init_state_equil(model, reg, G_ACC)
        b = init_state_equil(model, hand, G_ACC)
        assert np.allclose(a["pressure"], b["pressure"])
        assert np.allclose(a["sW"], b["sW"])
        print("ok test_factory_reproduces_hand_built_region")

    def test_two_regions_same_contact_equal_single():
        """Splitting the grid into two regions with the same contact reproduces
        the single-region whole-field equilibrium."""
        g, model = _model()
        nc = g.cells.num
        woc = 1050.0
        whole = get_initialization_regions_base_ad(model, _rho(model), [woc],
                                                datum_depth=1000.0,
                                                datum_pressure=200 * BARSA)
        full = init_state_equil(model, whole, G_ACC)

        half = nc // 2
        r1 = get_initialization_regions_base_ad(
            model, _rho(model), [woc], datum_depth=1000.0,
            datum_pressure=200 * BARSA, cells=np.arange(half))
        r2 = get_initialization_regions_base_ad(
            model, _rho(model), [woc], datum_depth=1000.0,
            datum_pressure=200 * BARSA, cells=np.arange(half, nc))
        multi = init_state_equil_regions(model, [r1, r2], G_ACC)
        assert np.allclose(multi["pressure"], full["pressure"])
        assert np.allclose(multi["sW"], full["sW"])
        print("ok test_two_regions_same_contact_equal_single")

    def test_two_regions_different_contacts_differ():
        """Two regions with different contacts give different saturations:
        region 1's contact is below the whole column (all oil), region 2's is
        above it (all water)."""
        g, model = _model()
        nc = g.cells.num
        half = nc // 2
        r1 = get_initialization_regions_base_ad(
            model, _rho(model), [1200.0], datum_depth=1000.0,   # WOC below -> oil
            datum_pressure=200 * BARSA, cells=np.arange(half))
        r2 = get_initialization_regions_base_ad(
            model, _rho(model), [900.0], datum_depth=1000.0,     # WOC above -> water
            datum_pressure=200 * BARSA, cells=np.arange(half, nc))
        st = init_state_equil_regions(model, [r1, r2], G_ACC)
        sw1 = st["sW"][:half].mean()
        sw2 = st["sW"][half:].mean()
        assert sw1 < 0.05 and sw2 > 0.95, (sw1, sw2)
        print("ok test_two_regions_different_contacts_differ")

    def test_min_max_saturations():
        g, model = _model()
        model.raw_fluid.swcon = 0.15
        s_min, s_max = get_min_max_phase_saturations_ad(model)
        assert s_min[0] == 0.15 and np.all(s_min <= s_max)
        print("ok test_min_max_saturations")

    test_factory_reproduces_hand_built_region()
    test_two_regions_same_contact_equal_single()
    test_two_regions_different_contacts_differ()
    test_min_max_saturations()
    print("\nAll tests passed!")
