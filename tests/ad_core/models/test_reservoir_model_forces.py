"""MRST `ReservoirModel`'s well/source/BC insertion group, ported.

This is the seam between *forces* and *equations*. The rest of the model builds
the reservoir's own residual and knows nothing about wells or boundaries; these
ten methods decide which equation each contribution lands in, with what sign,
and in what unit. All three are silent when wrong - the simulation still
converges, to the wrong answer - so each gets its own test here:

* **sign** - equations are residuals `accumulation + divergence - source = 0`,
  so sources are subtracted. Added instead, an injector drains the reservoir;
* **unit** - phase sources are mass rates divided by surface density (the
  equations are in surface volume), component sources are already mass and are
  not. Getting it backwards is a factor of ~800 for oil;
* **placement** - a per-cell source may only enter a per-cell equation, and
  boundary faces sharing a cell must be summed by the mapping matrix, not
  overwritten.

`test_the_surface_density_flip_is_a_no_op` guards something else: defining
`get_surface_densities` flips a `hasattr` probe in `boundary_conditions`, and
this is the third time in this port that adding a method has silently changed
which branch runs elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel

RHO_S = np.array([1000.0, 800.0, 1.2])


class _Model(ReservoirModel):
    """Phases named, operators absent - none of this group needs them."""

    def __init__(self, phases=("water", "oil"), n_cells=5, n_faces=8,
                 components=(), mass_basis=False):
        self.fluid = SimpleNamespace(phases=list(phases),
                                     n_phases=len(phases),
                                     rho_s=RHO_S[:len(phases)],
                                     disgas=False, vapoil=False)
        neighbors = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                              [-1, 0], [4, -1], [-1, 2], [2, -1]])
        self.G = SimpleNamespace(
            cells=SimpleNamespace(num=n_cells),
            faces=SimpleNamespace(num=n_faces, neighbors=neighbors),
            griddim=3)
        self.rock = SimpleNamespace(perm=np.full((n_cells, 1), 1e-13),
                                    poro=np.full(n_cells, 0.2))
        self.operators = SimpleNamespace()
        self.mass_basis = mass_basis
        self._components = list(components)
        self.FacilityModel = None
        self.outputFluxes = False
        self.groupings = []

    def get_component_names(self):
        return list(self._components)


def _src(cells, phase_mass, components=(), mapping=None):
    return SimpleNamespace(sourceCells=np.asarray(cells, dtype=int),
                           phaseMass=list(phase_mass),
                           phaseVolume=[np.zeros(len(cells))
                                        for _ in phase_mass],
                           components=list(components),
                           mapping=mapping)


# -- surface densities: the probe flip --------------------------------------

def test_the_surface_density_flip_is_a_no_op():
    """`boundary_conditions._surface_densities` probes for this method and
    until now fell through to `model.fluid.rho_s`. Defining it must return
    that exact array, or every existing BC computation changes silently."""
    from pymrst.modules.ad_core.utils.boundary_conditions import (
        _surface_densities,
    )
    model = _Model()
    np.testing.assert_array_equal(model.get_surface_densities(),
                                  np.asarray(model.fluid.rho_s).ravel())
    np.testing.assert_array_equal(_surface_densities(model),
                                  np.asarray(model.fluid.rho_s).ravel())


def test_surface_densities_can_be_restricted_to_chosen_phases():
    np.testing.assert_array_equal(
        _Model(("water", "oil", "gas")).get_surface_densities(phases=[0, 2]),
        RHO_S[[0, 2]])


# -- sign, the one that converges to the wrong answer -----------------------

def test_a_source_is_subtracted_from_the_residual():
    """`accumulation + divergence - source = 0`. Added instead, an injector
    drains the reservoir and the run still converges."""
    model = _Model()
    eqs = [np.full(5, 10.0), np.full(5, 10.0)]
    rates = [np.array([2000.0]), np.array([1600.0])]

    out = model.insert_sources(
        list(eqs), {"cells": [1], "value": rates})

    assert out[0][1] == 10.0 - 2000.0, "source must be subtracted"
    assert out[0][0] == 10.0, "untouched cells must not move"


def test_insert_sources_with_no_cells_changes_nothing():
    """Pins the property, not the mechanism.

    MRST guards this with an early return; here `subset_minus` is already a
    no-op on an empty subset, so deleting the guard changes nothing and no
    mutation of it is detectable. The property is still worth pinning - a
    schedule step with no sources must not perturb the residual - but the
    guard itself is untested on purpose, and `insert_sources` says so.
    """
    model = _Model()
    eqs = [np.full(5, 10.0)]
    # A stale value array left beside emptied cells, which is how this is
    # reached in practice: the struct is rebuilt each step, the cells first.
    out = model.insert_sources(eqs, {"cells": [], "value": [np.array([5.0])]})
    np.testing.assert_array_equal(out[0], np.full(5, 10.0))


# -- unit: phase sources scale, component sources do not --------------------

def test_phase_sources_are_divided_by_surface_density():
    """`phaseMass` is a mass rate; these equations are surface volume. The
    division is the unit conversion, and it is ~800x for oil."""
    model = _Model()
    facility = _facility(_src([1], [np.array([2000.0]), np.array([1600.0])]))
    model.FacilityModel = facility

    eqs, names, types, _, _ = model.insert_well_equations(
        [np.zeros(5), np.zeros(5)], ["water", "oil"], ["cell", "cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)

    assert eqs[0][1] == pytest.approx(-2000.0/RHO_S[0])
    assert eqs[1][1] == pytest.approx(-1600.0/RHO_S[1])


def test_component_sources_are_not_scaled():
    """Component equations are already in mass. Scaling them too would be
    dimensionally wrong in the same direction for every component, which is
    the kind of error that looks like a tuning problem."""
    model = _Model(components=("CO2",))
    model.FacilityModel = _facility(
        _src([2], [np.array([0.0]), np.array([0.0])],
             components=[np.array([7.0])]))

    eqs, *_ = model.insert_well_equations(
        [np.zeros(5), np.zeros(5), np.zeros(5)],
        ["water", "oil", "CO2"], ["cell", "cell", "cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)

    assert eqs[2][2] == pytest.approx(-7.0), "component source was rescaled"


# -- the well system is appended, not merged --------------------------------

def test_the_well_system_grows_the_equation_set():
    """Two things happen: sources are subtracted from existing equations, and
    the wells' own equations are appended as new rows."""
    model = _Model()
    model.FacilityModel = _facility(
        _src([1], [np.array([0.0]), np.array([0.0])]))

    eqs, names, types, _, src = model.insert_well_equations(
        [np.zeros(5), np.zeros(5)], ["water", "oil"], ["cell", "cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)

    assert names == ["water", "oil", "wellW", "wellO", "closureWells"]
    assert types == ["cell", "cell", "perf", "perf", "well"]
    assert len(eqs) == 5
    assert src is not None


def test_no_active_wells_leaves_the_system_untouched():
    model = _Model()
    model.FacilityModel = _facility(None, n_active=0)
    eqs, names, types, _, src = model.insert_well_equations(
        [np.zeros(5)], ["water"], ["cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)
    assert names == ["water"] and src is None and len(eqs) == 1


def test_an_equation_the_model_does_not_have_is_skipped():
    """A two-phase model has no "gas" equation. Absent must be a skip, not an
    error, or every non-three-phase model breaks."""
    model = _Model()
    model.FacilityModel = _facility(
        _src([1], [np.array([1.0]), np.array([1.0]), np.array([1.0])]))
    eqs, *_ = model.insert_well_equations(
        [np.zeros(5)], ["water"], ["cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)
    assert eqs[0][1] == pytest.approx(-1.0/RHO_S[0])


def test_equation_names_match_case_insensitively():
    """MRST uses `strcmpi` throughout; a model naming its equation "Water"
    must still receive water sources."""
    model = _Model()
    model.FacilityModel = _facility(
        _src([1], [np.array([2000.0]), np.array([0.0])]))
    eqs, *_ = model.insert_well_equations(
        [np.zeros(5)], ["WATER"], ["cell"],
        None, None, None, None, None, None, None, None, None, 1.0, None)
    assert eqs[0][1] == pytest.approx(-2000.0/RHO_S[0])


def test_a_source_may_not_enter_a_non_cell_equation():
    """Cell-indexed values into a well equation index rows meaning something
    else, and the system still solves."""
    model = _Model()
    model.FacilityModel = _facility(
        _src([1], [np.array([1.0]), np.array([1.0])]))
    with pytest.raises(ValueError, match="per cell"):
        model.insert_well_equations(
            [np.zeros(5)], ["water"], ["well"],
            None, None, None, None, None, None, None, None, None, 1.0, None)


# -- boundary flux signs ----------------------------------------------------

def test_boundary_flux_signs_follow_the_grid_not_the_boundary():
    """Boundary fluxes are positive *into* the reservoir; `state.flux` is
    positive from neighbour 0 to neighbour 1.

    Face 5 is `[4, -1]`: inward runs outside -> cell 4, i.e. from neighbour 1
    to neighbour 0, against the grid convention, so it flips. Face 4 is
    `[-1, 0]`: inward runs neighbour 0 -> neighbour 1 and they agree.

    Faces 4 and 6 have the outside first (keep); 5 and 7 have it second
    (flip). Both orientations are present so a constant sign cannot pass.
    """
    model = _Model()
    state = {"flux": np.zeros((8, 2))}
    forces = {"bc": {"face": np.array([4, 5, 6, 7])}}
    q = np.array([1.0, 1.0, 1.0, 1.0])

    out = model.store_boundary_fluxes(state, q, q, forces=forces)

    np.testing.assert_array_equal(out["flux"][[4, 5, 6, 7], 0],
                                  [1.0, -1.0, 1.0, -1.0])


def test_no_boundary_conditions_leaves_the_state_alone():
    model = _Model()
    state = {"flux": np.zeros((8, 2))}
    assert model.store_boundary_fluxes(state, None, None, forces={}) is state
    np.testing.assert_array_equal(state["flux"], 0.0)


# -- boundary conditions and sources ----------------------------------------

def test_several_boundary_faces_on_one_cell_are_summed():
    """The mapping matrix exists because boundary values are per *face* and
    the equation is per *cell*. Without it only the last face on each cell
    contributes and the rest are silently dropped."""
    model = _Model()
    mapping = sp.csr_array(np.array([[1.0, 1.0]]))  # two faces -> one cell
    bnd = _src([2], [np.array([300.0, 500.0]), np.array([0.0, 0.0])],
               mapping=mapping)
    eqs = _run_bc(model, bnd)
    assert eqs[0][2] == pytest.approx(-800.0/RHO_S[0]), "faces were not summed"


def test_a_mass_basis_model_does_not_rescale():
    """A `GenericReservoirModel` writes its equations in mass, so the surface
    density divisor is 1. Applying it anyway is the same ~800x error."""
    plain = _run_bc(_Model(), _src([2], [np.array([800.0]),
                                         np.array([0.0])]))
    mass = _run_bc(_Model(mass_basis=True),
                   _src([2], [np.array([800.0]), np.array([0.0])]))
    assert plain[0][2] == pytest.approx(-800.0/RHO_S[0])
    assert mass[0][2] == pytest.approx(-800.0)


def test_component_contributions_are_refused_on_the_base_class():
    """MRST raises rather than returning the equation untouched. Silently
    ignoring every component source converges to a wrong answer with no
    symptom; raising makes the missing override a failure."""
    with pytest.raises(NotImplementedError, match="CO2"):
        _Model().add_component_contributions("CO2", np.zeros(5), None,
                                             None, None)


# -- the small ones ---------------------------------------------------------

def test_cpr_scaling_is_one_factor_per_equation_name():
    scaling = _Model().get_scaling_factors_cpr(None, ["water", "oil"], None)
    assert scaling == [1, 1]


def test_the_extra_well_hooks_are_empty_and_consistent():
    """`get_extra_well_equation_names` sizes the system and
    `get_extra_well_contributions` fills it; a mismatch is a shape error at
    assembly, so they must agree even when both are empty."""
    model = _Model()
    names, types = model.get_extra_well_equation_names()
    eqs, src, eq_names = model.get_extra_well_contributions()
    assert len(names) == len(types) == len(eq_names) == len(eqs) == 0
    assert model.get_extra_well_primary_variable_names() == []


def test_setup_operators_uses_the_grid_it_is_given():
    """MRST allows a grid other than the model's, because some workflows
    discretize on a coarsened one while the model keeps the fine grid."""
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock

    model = _Model()
    g = compute_geometry(cart_grid([3, 3, 1], [30.0, 30.0, 10.0]))
    rock = make_rock(g, 100e-15, 0.2)

    returned = model.setup_operators(g, rock)
    assert returned is model, "the MRST spelling assigns the result"
    assert hasattr(model.operators, "T"), "operators were not built"
    assert np.asarray(model.operators.N).shape[0] > 0


# -- helpers ----------------------------------------------------------------

def _facility(src, n_active=1):
    wellsys = SimpleNamespace(
        wellEquations=[np.zeros(1), np.zeros(1)],
        names=["wellW", "wellO"], types=["perf", "perf"],
        controlEquation=np.zeros(1))
    return SimpleNamespace(
        get_number_of_active_wells=lambda ws=None: n_active,
        get_well_contributions=lambda *a, **k: (src, wellsys, None))


def _run_bc(model, bnd_cond):
    """Drive `add_boundary_conditions_and_sources` with a stubbed force
    computation, so the test is about placement rather than flux physics."""
    import pymrst.modules.ad_core.utils.boundary_conditions as bc_mod

    empty = _src([], [np.zeros(0), np.zeros(0)])
    original = bc_mod.compute_sources_and_boundary_conditions_ad_ad
    bc_mod.compute_sources_and_boundary_conditions_ad_ad = (
        lambda *a, **k: (empty, bnd_cond))
    try:
        eqs, _, _ = model.add_boundary_conditions_and_sources(
            [np.zeros(5), np.zeros(5)], ["water", "oil"], ["cell", "cell"],
            {}, None, [None, None], None, None, None, [], {})
        return eqs
    finally:
        bc_mod.compute_sources_and_boundary_conditions_ad_ad = original
