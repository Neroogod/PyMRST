"""MRST `ad-core/models/ReservoirModel.m` reservoir base model."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.components.components import Component, blackoil_components
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    FlowPropertyFunctions,
)
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    PVTPropertyFunctions,
    PhaseFluid,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value
from pymrst.utils.subset_arithmetic import subset_minus


class ReservoirModel(PhysicalModel):
    """A flow model on a grid: rock, fluid, operators, and state functions."""

    #: MRST `ReservoirModel.m` update limits. Only the saturation one is
    #: finite, and that asymmetry is the content: saturation is bounded and its
    #: properties are tabulated, so an over-long step lands off the end of a
    #: table, while pressure is unbounded and smooth and MRST declines to guess
    #: a scale for it. Declared here rather than inline in `__init__` so the
    #: values can be checked without constructing a model.
    UPDATE_LIMITS = {
        "dp_max_rel": float("inf"),
        "dp_max_abs": float("inf"),
        "ds_max_abs": 0.2,
        "minimum_pressure": float("-inf"),
        "maximum_pressure": float("inf"),
    }

    def __init__(self, g, rock, fluid: PhaseFluid, operators, pv_ref,
                 gdz: np.ndarray, gravity: float = 0.0,
                 components: Optional[List[Component]] = None,
                 mass_basis: bool = False):
        super().__init__()
        self.G = g
        self.rock = rock
        self.fluid = fluid
        self.operators = operators
        self.pv_ref = pv_ref
        self.gdz = gdz
        self.mass_basis = bool(mass_basis)
        self.gravity_magnitude = float(gravity)
        self.components = (components if components is not None
                           else blackoil_components(fluid.n_phases,
                                                    fluid.disgas,
                                                    fluid.vapoil))
        self.PVTPropertyFunctions = PVTPropertyFunctions(self)
        self.FlowPropertyFunctions = FlowPropertyFunctions(self)
        self.FlowDiscretization = FlowDiscretization(self)
        self.groupings = ["PVTPropertyFunctions", "FlowPropertyFunctions",
                          "FlowDiscretization"]
        # Read by the new `update_saturations`; nothing that existed before
        # consults them.
        for name, limit in self.UPDATE_LIMITS.items():
            setattr(self, name, limit)
        self.dpMaxRel = self.dp_max_rel
        self.dpMaxAbs = self.dp_max_abs
        self.dsMaxAbs = self.ds_max_abs

    @property
    def n_phases(self) -> int:
        return self.fluid.n_phases

    @property
    def n_components(self) -> int:
        return len(self.components)

    def component_names(self) -> List[str]:
        return [c.name for c in self.components]

    def prepare_report_step(self, state, state0, dt: float, forces: Any = None):
        """MRST ``ReservoirModel.prepareReportstep``: forward to the facility.

        MRST guards this on `drivingForces.W` being non-empty; here the guard
        is the facility having wells at all, which is the same question asked
        of the object that actually holds them.

        The facility is read from `facility` first and `FacilityModel` second.
        Both names exist on this port, and it is the lower-case one the models
        actually populate - `FacilityModel` is None on a built model, so
        looking only there would silently forward to nothing.
        """
        self, state = super().prepare_report_step(state, state0, dt, forces)
        facility = getattr(self, "facility", None)
        if facility is None:
            facility = getattr(self, "FacilityModel", None)
        if facility is not None and getattr(facility, "wells", None):
            # MRST attaches the reservoir model to the facility inside
            # `validateModel` (ReservoirModel.m:212), which `prepareReportstep`
            # calls. This port only does it when a model is handed to
            # `FacilityModel.__init__`, and the models do not hand one over -
            # so anything on the facility that reads `ReservoirModel` (the
            # reservoir-volume conversion does, for PVT regions and phase
            # indices) would raise `AttributeError` on first use.
            if getattr(facility, "ReservoirModel", None) is None:
                facility.set_reservoir_model(self)
            facility, state = facility.prepare_report_step(
                state, state0, dt, forces)
        return self, state

    prepareReportstep = prepare_report_step


    def prepare_timestep(self, state, state0, dt: float, forces: Any = None):
        """MRST ``ReservoirModel.prepareTimestep``: forward to the facility
        (ReservoirModel.m:183). This is where a rate-controlled well with a
        zero target gets marked shut, so it must run every ministep - not once
        per report step like `prepare_report_step`."""
        self, state = super().prepare_timestep(state, state0, dt, forces)
        facility = getattr(self, "facility", None) or getattr(
            self, "FacilityModel", None)
        if facility is not None and getattr(facility, "wells", None):
            if getattr(facility, "ReservoirModel", None) is None:
                facility.set_reservoir_model(self)
            prepare = getattr(facility, "prepare_timestep", None)
            if prepare is not None:
                facility, state = prepare(state, state0, dt, forces)
        return self, state

    prepareTimestep = prepare_timestep

    def get_component_names(self) -> List[str]:
        """Return component names using MRST's ``getComponentNames`` surface."""
        return self.component_names()

    getComponentNames = get_component_names

    def get_number_of_components(self) -> int:
        return self.n_components

    getNumberOfComponents = get_number_of_components

    @staticmethod
    def _mrst_phase_name(name: str) -> str:
        lookup = {"water": "W", "oil": "O", "gas": "G"}
        key = str(name).lower()
        return lookup.get(key, str(name).upper())

    def get_phase_names(self) -> List[str]:
        """Return active phase names in MRST's short-name convention."""
        return [self._mrst_phase_name(name) for name in self.fluid.phases]

    getPhaseNames = get_phase_names

    def get_number_of_phases(self) -> int:
        return self.n_phases

    getNumberOfPhases = get_number_of_phases

    def get_active_phases(self) -> np.ndarray:
        """Return a boolean mask for active phases, matching MRST helper API."""
        return np.ones(self.n_phases, dtype=bool)

    getActivePhases = get_active_phases

    def get_phase_index(self, name: str) -> int:
        """Return the zero-based phase index for an MRST phase name."""
        key = str(name).lower()
        short = [str(item).lower() for item in self.get_phase_names()]
        long = [str(item).lower() for item in self.fluid.phases]
        if key in short:
            return short.index(key)
        if key in long:
            return long.index(key)
        raise ValueError(f"unknown phase name {name!r}")

    getPhaseIndex = get_phase_index

    @property
    def component_rho_s(self) -> np.ndarray:
        rho_s = np.ones(self.n_components)
        n = min(self.n_components, self.fluid.n_phases)
        rho_s[:n] = self.fluid.rho_s[:n]
        return rho_s

    def _basis_scale(self) -> np.ndarray:
        if not self.mass_basis:
            return np.ones(self.n_components)
        return self.component_rho_s

    def component_state_keys(self) -> List[str]:
        keys = ["rs"] if self.fluid.disgas else []
        keys += ["rv"] if self.fluid.vapoil else []
        keys += [c.state_key for c in self.components
                 if hasattr(c, "state_key")]
        return keys

    def conservation_equations(self, state0: State, state: State, dt: float):
        return self.FlowDiscretization.component_conservation_equations(
            self, state, state0, dt
        )

    # ======================================================================
    # MRST `ReservoirModel.m`, ported (Phase 2). Everything below is new;
    # nothing in PyMRST called any of it before.
    # ======================================================================

    # -- phase presence -----------------------------------------------------
    #
    # MRST keeps three flags and indexes the fixed [water, oil, gas] triple.
    # PyMRST's `get_active_phases` returns a mask of length `n_phases` instead,
    # so for a two-phase oil-water model MRST says `[1 1 0]` and PyMRST says
    # `[1 1]` - the same name for two different things, which coincide only on
    # a three-phase model. Callers already assume MRST's shape (see
    # `tests/ad_core/utils/test_well_vector_utilities.py`, which hands in
    # `[False, True, True]`).
    #
    # Changing `get_active_phases` is a behaviour change and belongs to the
    # migration phase, so the triple is exposed separately and every method
    # below uses it. The divergence is recorded rather than papered over.

    # MRST spells these `model.water` / `.oil` / `.gas`, and this port did
    # expose them that way for one revision. That was a behaviour change, not
    # an addition: roughly ten call sites across ad_props, compositional and
    # ad_core/initialization probe `getattr(model, "water", False)`, and since
    # no concrete model defined the attribute, every one of them had been
    # receiving `False`. Adding the property silently flipped them to the real
    # value - changing which branches run, in code with no test coverage on
    # those paths.
    #
    # So the triple is computed rather than exposed. Restoring MRST's names is
    # a migration-phase decision, and it needs those call sites reviewed first,
    # because several of them are almost certainly relying on the `False` they
    # have always been given.

    def get_phase_presence(self) -> np.ndarray:
        """MRST's `[water, oil, gas]` triple, whatever the active count."""
        names = set(self.get_phase_names())
        return np.array([n in names for n in ("W", "O", "G")], dtype=bool)

    getPhasePresence = get_phase_presence

    def get_phase_indices(self):
        """`(iW, iO, iG)`, one-based into the active phases, `-1` when absent.

        MRST's cumulative-count trick: a phase's index is how many active
        phases precede it, so removing water shifts oil and gas down by one.
        """
        present = self.get_phase_presence()
        w, o, g = (int(present[0]), int(present[1]), int(present[2]))
        indices = np.array([w, w + o, w + o + g], dtype=np.int64)
        indices[~self.get_phase_presence()] = -1
        return tuple(int(i) for i in indices)

    getPhaseIndices = get_phase_indices

    def get_saturation_var_names(self) -> List[str]:
        """`['sw', 'so', 'sg']`, restricted to the phases that exist."""
        return [name for name, present
                in zip(("sw", "so", "sg"), self.get_phase_presence()) if present]

    getSaturationVarNames = get_saturation_var_names

    def sat_var_index(self, name: str) -> int:
        """Zero-based index of a saturation variable, or -1.

        MRST returns an empty index for an unknown name; -1 is the Python
        spelling of "not there", and is checked rather than used blindly.
        """
        names = [n.lower() for n in self.get_saturation_var_names()]
        key = str(name).lower()
        return names.index(key) if key in names else -1

    satVarIndex = sat_var_index

    def comp_var_index(self, name: str) -> int:
        names = [str(n).lower() for n in self.component_names()]
        key = str(name).lower()
        return names.index(key) if key in names else -1

    compVarIndex = comp_var_index

    def split_primary_variables(self, variables: Sequence[str]):
        """Split into `(rest, saturations, wells)`.

        The saturation block is treated differently by every update rule - it
        is bounded and it sums to one - so it has to be separable from the
        rest before any of them can be applied.
        """
        sat_names = {n.lower() for n in self.get_saturation_var_names()}
        facility = getattr(self, "FacilityModel", None)
        well_names = set()
        if facility is not None:
            getter = getattr(facility, "get_primary_variable_names",
                             getattr(facility, "getPrimaryVariableNames", None))
            if getter is not None:
                well_names = {str(n).lower() for n in getter()}

        saturations = [v for v in variables if str(v).lower() in sat_names]
        wells = [v for v in variables if str(v).lower() in well_names]
        rest = [v for v in variables
                if str(v).lower() not in sat_names
                and str(v).lower() not in well_names]
        return rest, saturations, wells

    splitPrimaryVariables = split_primary_variables

    # -- saturation updates -------------------------------------------------

    @staticmethod
    def adjust_step_from_sat_bounds(s, ds):
        """Shorten an update so no saturation leaves `[0, 1]`.

        Not a clamp. Clamping after the fact would move one phase and leave the
        others, so the saturations would no longer sum to one; this scales the
        *whole row* by the factor that brings the worst offender exactly to its
        bound, which shortens the step without changing its direction.
        """
        s = np.atleast_2d(np.asarray(value(s), dtype=float))
        ds = np.atleast_2d(np.asarray(value(ds), dtype=float)).copy()
        candidate = s + ds

        over = np.maximum(candidate - 1.0, 0.0)
        under = np.abs(np.minimum(candidate, 0.0))
        violation = np.maximum(over, under)

        worst = violation.max(axis=1)
        which = violation.argmax(axis=1)
        bad = worst > 0.0
        if not np.any(bad):
            return ds

        weight = np.ones(s.shape[0])
        rows = np.flatnonzero(bad)
        offending = np.abs(ds[rows, which[rows]])
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = (offending - worst[rows])/offending
        weight[rows] = np.where(np.isfinite(scaled), scaled, 1.0)
        ds[bad, :] = ds[bad, :]*weight[bad, None]
        return ds

    adjustStepFromSatBounds = adjust_step_from_sat_bounds

    def update_saturations(self, state: State, dx, problem,
                           sat_vars: Optional[Sequence[str]] = None) -> State:
        """Apply the saturation increments, filling the dependent phase.

        Only `n-1` saturations are solved for; the last is `1 - sum(rest)`, and
        that is what makes the set consistent. Solving for all of them and
        renormalising afterwards is a different scheme.
        """
        if sat_vars is None:
            _, sat_vars, _ = self.split_primary_variables(
                getattr(problem, "primary_variables", None)
                or getattr(problem, "var_names", None) or [])
        if not sat_vars:
            return state

        names = [n.lower() for n in self.get_saturation_var_names()]
        solved = {str(v).lower() for v in sat_vars}
        fill = [n for n in names if n not in solved]
        if len(fill) > 1:
            raise ValueError(
                f"{len(fill)} saturations are unsolved for ({fill}); exactly "
                f"one may be the dependent phase")

        n_cells = int(self.G.cells.num)
        ds = np.zeros((n_cells, len(names)))
        remainder = np.zeros(n_cells)
        for i, name in enumerate(names):
            if name in solved:
                increment = np.asarray(
                    value(self.get_increment(dx, problem, name)), dtype=float)
                ds[:, i] = increment
                if fill:
                    remainder = remainder - increment
        if fill:
            ds[:, names.index(fill[0])] = remainder

        ds = self.adjust_step_from_sat_bounds(np.asarray(state["s"]), ds)
        state, _, _ = self.update_state_from_increment(
            state, ds, problem, "s", None,
            getattr(self, "ds_max_abs", getattr(self, "dsMaxAbs", None)))

        saturations = np.asarray(value(state["s"]), dtype=float)
        if fill:
            bad = np.any((saturations > 1.0) | (saturations < 0.0), axis=1)
            if np.any(bad):
                rows = np.clip(saturations[bad, :], 0.0, 1.0)
                totals = rows.sum(axis=1, keepdims=True)
                # A row that clamped to all-zero has no direction left to
                # normalise; leaving it alone beats dividing by zero.
                rows = np.divide(rows, totals, out=rows,
                                 where=totals > 0.0)
                saturations = saturations.copy()
                saturations[bad, :] = rows
        else:
            saturations = np.maximum(saturations, 0.0)
        state["s"] = saturations
        return state

    updateSaturations = update_saturations

    # -- relative permeability ----------------------------------------------

    @staticmethod
    def rel_perm_wo(sw, so, f, **kwargs):
        kr_o = getattr(f, "kr_o", None) or getattr(f, "krO", None)
        if kr_o is None:
            kr_o = getattr(f, "kr_ow", None) or getattr(f, "krOW")
        kr_w = getattr(f, "kr_w", None) or getattr(f, "krW")
        return kr_w(sw, **kwargs), kr_o(so, **kwargs)

    @staticmethod
    def rel_perm_og(so, sg, f, **kwargs):
        kr_o = getattr(f, "kr_o", None) or getattr(f, "krO", None)
        if kr_o is None:
            kr_o = getattr(f, "kr_og", None) or getattr(f, "krOG")
        kr_g = getattr(f, "kr_g", None) or getattr(f, "krG")
        return kr_o(so, **kwargs), kr_g(sg, **kwargs)

    @staticmethod
    def rel_perm_wg(sw, sg, f, **kwargs):
        kr_w = getattr(f, "kr_w", None) or getattr(f, "krW")
        kr_g = getattr(f, "kr_g", None) or getattr(f, "krG")
        return kr_w(sw, **kwargs), kr_g(sg, **kwargs)

    @staticmethod
    def rel_perm_wog(sw, so, sg, f, **kwargs):
        """Three-phase oil relative permeability by saturation-weighted
        interpolation between the two-phase curves.

        The oil curve is not measured in three-phase conditions; it is
        interpolated between the oil-water and oil-gas curves by how much of
        the non-oil pore space each of water and gas occupies. `swcon` is the
        connate water, which does not participate, and is clipped just below
        `sw` so the denominator cannot reach zero when a cell is at connate.
        """
        kr_w = getattr(f, "kr_w", None) or getattr(f, "krW")
        kr_g = getattr(f, "kr_g", None) or getattr(f, "krG")
        kr_ow = getattr(f, "kr_ow", None) or getattr(f, "krOW")
        kr_og = getattr(f, "kr_og", None) or getattr(f, "krOG")

        swcon = getattr(f, "swcon", getattr(f, "sWcon", 0.0)) or 0.0
        swcon = np.minimum(swcon, np.asarray(value(sw)) - 1.0e-5)

        denominator = sg + sw - swcon
        weight_w = (sw - swcon)/denominator
        weight_g = 1.0 - weight_w
        return (kr_w(sw, **kwargs),
                weight_g*kr_og(so, **kwargs) + weight_w*kr_ow(so, **kwargs),
                kr_g(sg, **kwargs))

    relPermWO = rel_perm_wo
    relPermOG = rel_perm_og
    relPermWG = rel_perm_wg
    relPermWOG = rel_perm_wog

    def evaluate_rel_perm(self, sat: Sequence[Any], **kwargs):
        """Relative permeabilities for the active phases, in phase order.

        Dispatches on which phases exist, exactly as MRST does by building the
        method name from the phase letters.
        """
        names = "".join(self.get_phase_names())
        if len(sat) != len(names):
            raise ValueError(
                f"{len(sat)} saturations for {len(names)} active phases")
        if len(names) == 1:
            fluid = getattr(self, "raw_fluid", self.fluid)
            kr = (getattr(fluid, f"kr_{names.lower()}", None)
                  or getattr(fluid, f"kr{names}"))
            return (kr(sat[0], **kwargs),)
        method = getattr(self, f"relPerm{names}", None)
        if method is None:
            raise NotImplementedError(
                f"no relative-permeability rule for phase set {names!r}")
        fluid = getattr(self, "raw_fluid", self.fluid)
        return method(*sat, fluid, **kwargs)

    evaluateRelPerm = evaluate_rel_perm

    # -- gravity ------------------------------------------------------------

    def get_gravity_vector(self) -> np.ndarray:
        """The gravity vector, truncated to the grid's dimension."""
        gravity = getattr(self, "gravity", None)
        if gravity is None or np.ndim(gravity) == 0:
            magnitude = float(self.gravity_magnitude)
            dim = int(getattr(self.G, "griddim", 3))
            vector = np.zeros(dim)
            vector[-1] = magnitude
            return vector
        vector = np.asarray(gravity, dtype=float).ravel()
        dim = int(getattr(self.G, "griddim", vector.size))
        return vector[:dim]

    getGravityVector = get_gravity_vector

    def get_gravity_gradient(self) -> np.ndarray:
        """`grad(z) . g` per interface - the gravity term of the flux.

        Taken from `operators.gdz` when the discretization already built it,
        because recomputing it from centroids would silently disagree with the
        operator the fluxes actually use on any grid where the two differ.
        """
        gdz = getattr(self.operators, "gdz", None)
        if gdz is None and isinstance(self.operators, dict):
            gdz = self.operators.get("gdz")
        if gdz is not None:
            return np.asarray(gdz, dtype=float)
        if self.gdz is not None:
            return np.asarray(self.gdz, dtype=float)
        centroids = getattr(self.G.cells, "centroids", None)
        if centroids is None:
            raise ValueError(
                "grid has no cell centroids; run compute_geometry first")
        grad = getattr(self.operators, "Grad", None)
        if grad is None:
            raise ValueError("operators have no Grad; cannot form gdz")
        return np.asarray(grad(centroids) @ self.get_gravity_vector())

    getGravityGradient = get_gravity_gradient

    # -- storing computed phase data on the state ---------------------------

    def set_phase_data(self, state: State, data: Sequence[Any], field: str,
                       subs=None) -> State:
        """Write per-phase values into a `(n, n_active)` field.

        `data` is always the full `[W, O, G]` triple; the inactive entries are
        skipped and the active ones written in order, so a two-phase model
        writes two columns from a three-element list.
        """
        presence = self.get_phase_presence()
        target = np.asarray(state[field])
        column = 0
        for i, present in enumerate(presence):
            if not present:
                continue
            values = np.asarray(value(data[i]), dtype=float).ravel()
            if subs is None:
                target[:, column] = values
            else:
                target[np.asarray(subs), column] = values
            column += 1
        state[field] = target
        return state

    setPhaseData = set_phase_data

    def _store_cell_field(self, state, field, data, dtype=float):
        n_active = int(self.get_phase_presence().sum())
        state[field] = np.zeros((int(self.G.cells.num), n_active), dtype=dtype)
        return self.set_phase_data(state, data, field)

    def store_mobilities(self, state, mob_w=None, mob_o=None, mob_g=None):
        return self._store_cell_field(state, "mob", (mob_w, mob_o, mob_g))

    def store_density(self, state, rho_w=None, rho_o=None, rho_g=None):
        return self._store_cell_field(state, "rho", (rho_w, rho_o, rho_g))

    def store_bfactors(self, state, b_w=None, b_o=None, b_g=None):
        return self._store_cell_field(state, "bfactor", (b_w, b_o, b_g))

    def store_fluxes(self, state, v_w=None, v_o=None, v_g=None):
        """Phase fluxes, written onto the *internal* connections only.

        MRST sizes `state.flux` by the internal connection count and indexes it
        with that same mask, so a boundary face never receives a value.
        """
        internal = getattr(self.operators, "internalConn", None)
        if internal is None and isinstance(self.operators, dict):
            internal = self.operators.get("internalConn")
        n_active = int(self.get_phase_presence().sum())
        if internal is None:
            n_faces = int(self.G.faces.num)
            state["flux"] = np.zeros((n_faces, n_active))
            return self.set_phase_data(state, (v_w, v_o, v_g), "flux")
        internal = np.asarray(internal)
        state["flux"] = np.zeros((internal.size, n_active))
        return self.set_phase_data(state, (v_w, v_o, v_g), "flux",
                                   subs=np.flatnonzero(internal)
                                   if internal.dtype == bool else internal)

    def store_upstream_indices(self, state, up_w=None, up_o=None, up_g=None):
        neighbors = getattr(self.operators, "N", None)
        if neighbors is None and isinstance(self.operators, dict):
            neighbors = self.operators.get("N")
        n_interfaces = (np.asarray(neighbors).shape[0] if neighbors is not None
                        else int(self.G.faces.num))
        n_active = int(self.get_phase_presence().sum())
        state["upstreamFlag"] = np.zeros((n_interfaces, n_active), dtype=bool)
        return self.set_phase_data(state, (up_w, up_o, up_g), "upstreamFlag")

    storeMobilities = store_mobilities
    storeDensity = store_density
    storebfactors = store_bfactors
    storeFluxes = store_fluxes
    storeUpstreamIndices = store_upstream_indices

    # ======================================================================
    # Wells, sources and boundary conditions.
    #
    # This is the group that turns *forces* into *equations*. Everything above
    # builds the reservoir's own residual; nothing above knows how a well or a
    # boundary enters it. MRST keeps that seam deliberately: the flux
    # computations live in `computeSourcesAndBoundaryConditionsAD` and the
    # `FacilityModel`, and these ten methods are the part that decides which
    # equation each contribution lands in, with what sign, and in what unit.
    #
    # The sign is the same throughout and it is worth stating once. Equations
    # are written as residuals `accumulation + divergence - source = 0`, so a
    # source term is *subtracted*. An injector has a positive rate and reduces
    # the residual; a producer's is negative and raises it. Adding instead of
    # subtracting produces a converging simulation of the opposite problem.
    # ======================================================================

    def get_surface_densities(self, regions=None, phases=None) -> np.ndarray:
        """Surface densities of the active phases, in canonical WOG order.

        MRST reads `fluid.rhoWS`/`rhoOS`/`rhoGS` per phase; PyMRST's fluids
        carry the same numbers as `fluid.rho_s`, already in phase order.

        Defining this method flips a probe: `boundary_conditions.
        _surface_densities` does `hasattr(model, "get_surface_densities")` and
        until now fell through to `model.fluid.rho_s`. The flip is a no-op
        *by construction* - called with no arguments this returns that exact
        array - and `test_surface_densities_flip_is_a_no_op` pins it. That is
        the only reason this method is safe to add to a class that is already
        in use, and the reason it is not written any other way.
        """
        rho_s = np.asarray(self.fluid.rho_s, dtype=float).ravel()
        if phases is not None:
            rho_s = rho_s[np.asarray(phases)]
        if regions is not None:
            rho_s = np.atleast_2d(rho_s)[np.asarray(regions), :]
        return rho_s

    getSurfaceDensities = get_surface_densities

    # -- operators ----------------------------------------------------------

    def setup_operators(self, G=None, rock=None, **kwargs) -> "ReservoirModel":
        """Build the discrete operators (MRST `setupOperators`).

        MRST allows a grid and rock other than the model's own, because some
        workflows discretize on a coarsened or otherwise modified grid while
        the model keeps the fine one. Both default to the model's.

        MRST returns a modified copy (`model = model.setupOperators(...)`);
        Python mutates in place. `self` is returned anyway so that the MRST
        spelling of the call reads correctly and assigns the same object,
        rather than silently assigning `None`.
        """
        from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa

        grid = self.G if G is None else G
        rock_ = self.rock if rock is None else rock
        self.operators = setup_operators_tpfa(grid, rock_, **kwargs)
        return self

    setupOperators = setup_operators

    # -- CPR scaling --------------------------------------------------------

    def get_scaling_factors_cpr(self, problem=None, names=None,
                                solver=None) -> List[Any]:
        """Per-equation scaling for CPR reduction; unity on the base class.

        MRST's base implementation returns one `1` per name and expects
        subclasses to override with something meaningful (the black-oil models
        return `1/b` weights). Returning unity is not a stub: it is the
        identity scaling, and a model that does not know better must not
        invent weights - wrong CPR weights do not fail, they just make the
        pressure system stop being elliptic and the iteration count explode.
        """
        return [1 for _ in (names or [])]

    getScalingFactorsCPR = get_scaling_factors_cpr

    # -- boundary fluxes into the state -------------------------------------

    def store_boundary_fluxes(self, state: State, q_w=None, q_o=None,
                              q_g=None, forces=None) -> State:
        """Write integrated boundary phase fluxes onto their faces.

        The sign correction is the whole content of this method. Boundary
        fluxes are computed positive *into* the reservoir, but `state.flux`
        follows the grid's own convention: positive from `neighbors[:, 0]` to
        `neighbors[:, 1]`.

        So the two agree only when the outside is the *first* neighbour - then
        "inward" and "0 to 1" point the same way. When the outside is the
        second neighbour, inward runs from 1 to 0, against the convention, and
        the value must be negated. That is MRST's
        `sgn = 1 - 2*(neighbors(faces, 2) == 0)`.

        Skipping this leaves fluxes that are individually plausible and
        collectively non-conservative, on exactly the boundary faces where the
        grid happens to be numbered the other way round.
        """
        bc = _force_field(forces, "bc")
        if bc is None:
            return state
        faces = np.asarray(_force_field(bc, "face", []), dtype=int).ravel()
        if faces.size == 0:
            return state

        neighbors = np.asarray(self.G.faces.neighbors)
        # PyMRST stores neighbours 0-based with -1 for "outside"; MRST uses
        # 1-based with 0. Testing "is there no cell on side 2" has to be
        # written in this port's own convention, and a copied `== 0` would
        # silently mean "is cell zero", which is a real cell.
        outside = -1 if neighbors.min() < 0 else 0
        sign = 1 - 2*(neighbors[faces, 1] == outside)

        fluxes = []
        for q in (q_w, q_o, q_g):
            if q is None:
                fluxes.append(None)
            else:
                fluxes.append(np.asarray(value(q), dtype=float).ravel()*sign)

        if "flux" not in state or np.asarray(state["flux"]).ndim != 2:
            n_active = int(self.get_phase_presence().sum())
            state["flux"] = np.zeros((int(self.G.faces.num), n_active))
        return self.set_phase_data(state, fluxes, "flux", subs=faces)

    storeBoundaryFluxes = store_boundary_fluxes

    # -- wells --------------------------------------------------------------

    def insert_well_equations(self, eqs, names, types, well_sol0, well_sol,
                              well_vars, well_map, p, mob, rho, dissolved,
                              components, dt, opt=None):
        """Add well sources to the reservoir equations and append the well
        system (MRST `insertWellEquations`).

        Two distinct things happen, and conflating them is the usual error.
        The wells' *source terms* are subtracted from the existing reservoir
        equations at the perforated cells; the wells' *own* equations (one per
        phase per well, plus one control equation) are appended as new rows.
        The system grows in both directions.

        Phase sources are divided by the surface density and component sources
        are not. That asymmetry is not arbitrary: `phaseMass` is a mass rate
        and the phase equations here are written in surface volume, so the
        division is the unit conversion between them. Component equations are
        already in mass. Dividing both, or neither, gives a residual that is
        dimensionally inconsistent by a factor of ~800 for oil.
        """
        facility = self.FacilityModel
        if facility is None or facility.get_number_of_active_wells(well_sol) == 0:
            return list(eqs), list(names), list(types), well_sol, None

        iteration = getattr(opt, "iteration", None)
        if iteration is None and isinstance(opt, dict):
            iteration = opt.get("iteration")
        src, wellsys, well_sol = facility.get_well_contributions(
            well_sol0, well_sol, well_vars, well_map, p, mob, rho,
            dissolved, components, dt, iteration)

        eqs, names, types = list(eqs), list(names), list(types)
        rho_s = self.get_surface_densities()
        wc = np.asarray(_force_field(src, "sourceCells", []), dtype=int).ravel()

        long_names = [_LONG_PHASE_NAMES.get(n.upper(), n)
                      for n in self.get_phase_names()]
        phase_mass = _force_field(src, "phaseMass", []) or []
        for i, long in enumerate(long_names):
            sub = _index_of(names, long)
            if sub is None or i >= len(phase_mass):
                continue
            _assert_cell_type(types, sub, long)
            eqs[sub] = subset_minus(eqs[sub], phase_mass[i]/rho_s[i], wc)

        comp_src = _force_field(src, "components", []) or []
        for i, cname in enumerate(self.get_component_names()):
            sub = _index_of(names, cname)
            if sub is None or i >= len(comp_src):
                continue
            _assert_cell_type(types, sub, cname)
            eqs[sub] = subset_minus(eqs[sub], comp_src[i], wc)

        well_eqs = list(_force_field(wellsys, "wellEquations", []) or [])
        eqs = eqs + well_eqs + [_force_field(wellsys, "controlEquation")]
        names = names + list(_force_field(wellsys, "names", []) or []) \
            + ["closureWells"]
        types = types + list(_force_field(wellsys, "types", []) or []) \
            + ["well"]
        return eqs, names, types, well_sol, src

    insertWellEquations = insert_well_equations

    def get_extra_well_contributions(self, well=None, well_sol0=None,
                                     well_sol=None, q_s=None, bh=None,
                                     packed=None, q_mass=None, q_vol=None,
                                     dt=None, iteration=None):
        """Extra well equations and component sources; none on the base class.

        The hook exists so that a model with components needing special well
        treatment (polymer, surfactant, thermal) can add closure equations
        without the well model knowing what they are. Empty here is the
        correct answer, not a placeholder: a plain reservoir model contributes
        nothing beyond the phase sources already handled.
        """
        return [], [], []

    getExtraWellContributions = get_extra_well_contributions

    def get_extra_well_equation_names(self):
        """`(names, types)` of extra well equations. Must stay consistent with
        `get_extra_well_contributions` - the well model sizes its system from
        this and fills it from that, so a mismatch is a shape error at
        assembly rather than anything diagnosable."""
        return [], []

    getExtraWellEquationNames = get_extra_well_equation_names

    def get_extra_well_primary_variable_names(self) -> List[str]:
        """Extra well primary variables; none on the base class."""
        return []

    getExtraWellPrimaryVariableNames = get_extra_well_primary_variable_names

    # -- boundary conditions and sources ------------------------------------

    def add_boundary_conditions_and_sources(self, eqs, names, types, state,
                                            p, s, mob, rho, dissolved,
                                            components, forces):
        """Add BC and source contributions to the equations (MRST
        `addBoundaryConditionsAndSources`).

        Handles both ways a model can be written, which is why it looks
        redundant. A black-oil-style model names its equations after phases
        ("water", "oil", "gas") and gets phase sources directly. A
        compositional model names them after components and takes the second
        loop instead. MRST warns that mixing the two gives unexpected source
        terms, and this port does not try to be cleverer about it.

        The `scale` divisor carries the same unit question as
        `insert_well_equations`: a `GenericReservoirModel` writes its equations
        in mass and divides by 1, anything else writes surface volume and
        divides by the surface density.

        Boundary contributions may carry a `mapping` matrix. It exists because
        several boundary *faces* can belong to one *cell*, and the fluxes are
        per face while the equation is per cell - the matrix sums them. Using
        the face values directly would add only the last face on each cell.
        """
        from pymrst.modules.ad_core.utils.boundary_conditions import (
            compute_sources_and_boundary_conditions_ad_ad as compute_sources_and_boundary_conditions_ad,
        )

        src_terms, bnd_cond = compute_sources_and_boundary_conditions_ad(
            self, p, s, mob, rho, dissolved, forces)
        eqs = list(eqs)
        rho_s = self.get_surface_densities()
        long_names = [_LONG_PHASE_NAMES.get(n.upper(), n)
                      for n in self.get_phase_names()]
        is_mass = bool(getattr(self, "mass_basis", False))

        bc_force = _force_field(forces, "bc")
        for i in range(len(s)):
            if i >= len(long_names):
                break
            sub = _index_of(names, long_names[i])
            if sub is None:
                continue
            _assert_cell_type(types, sub, long_names[i])
            scale = 1.0 if is_mass else rho_s[i]

            sc = np.asarray(_force_field(src_terms, "sourceCells", []),
                            dtype=int).ravel()
            if sc.size:
                eqs[sub] = subset_minus(
                    eqs[sub], src_terms.phaseMass[i]/scale, sc)

            # A caller may supply boundary phase masses directly rather than
            # having them computed - MRST's `forces.bc.phaseMass` override.
            supplied = _force_field(bc_force, "phaseMass")
            if supplied is not None:
                bnd_cond.phaseMass[i] = np.asarray(supplied)[:, i]

            bcc = np.asarray(_force_field(bnd_cond, "sourceCells", []),
                             dtype=int).ravel()
            if bcc.size:
                q = bnd_cond.phaseMass[i]
                mapping = _force_field(bnd_cond, "mapping")
                if mapping is not None:
                    q = mapping @ q
                eqs[sub] = subset_minus(eqs[sub], q/scale, bcc)

        if getattr(self, "outputFluxes", False):
            presence = self.get_phase_presence()
            volumes = [None, None, None]
            active = [i for i, on in enumerate(presence) if on]
            for slot, vol in zip(active, bnd_cond.phaseVolume):
                volumes[slot] = vol
            state = self.store_boundary_fluxes(state, *volumes, forces=forces)

        for i, cname in enumerate(self.get_component_names()):
            if cname in long_names:
                continue  # already handled as a phase pseudocomponent
            sub = _index_of(names, cname)
            if sub is None:
                eq = np.zeros(int(self.G.cells.num))
            else:
                _assert_cell_type(types, sub, cname)
                eq = eqs[sub]
            C = components[i] if i < len(components) else None
            eq, bnd_cond = self.add_component_contributions(
                cname, eq, C, bnd_cond, _force_field(forces, "bc"))
            eq, src_terms = self.add_component_contributions(
                cname, eq, C, src_terms, _force_field(forces, "src"))
            if sub is not None:
                eqs[sub] = eq

        return eqs, state, {"src": src_terms, "bc": bnd_cond}

    addBoundaryConditionsAndSources = add_boundary_conditions_and_sources

    def add_component_contributions(self, cname, eq, component, src, force):
        """Component source terms; not valid on the base class.

        MRST raises here rather than returning `eq` unchanged, and the
        distinction matters. A base class that silently returned the equation
        untouched would let a component model with a missing override run to
        convergence having quietly ignored every component source - a wrong
        answer with no symptom. Raising makes the omission a failure.
        """
        raise NotImplementedError(
            f"add_component_contributions is not valid on the base class "
            f"(asked for component {cname!r}); a model with components must "
            f"implement it")

    addComponentContributions = add_component_contributions

    def insert_sources(self, eqs, src):
        """Subtract pre-computed source values from each equation.

        The empty-cells early return is MRST's and is kept for structural
        fidelity, but it is an optimization here rather than a guard: PyMRST's
        `subset_minus` is already a no-op on an empty subset, including when a
        stale non-empty value array is left beside it. Removing it changes no
        observable behaviour, so no test covers it - stated here so that the
        absence of a test is a finding rather than an oversight.
        """
        cells = np.asarray(_force_field(src, "cells", []), dtype=int).ravel()
        if cells.size == 0:
            return eqs
        values = _force_field(src, "value", []) or []
        eqs = list(eqs)
        for i in range(len(eqs)):
            if i < len(values):
                eqs[i] = subset_minus(eqs[i], values[i], cells)
        return eqs

    insertSources = insert_sources


#: MRST names its equations with the long phase names; `get_phase_names`
#: returns the short ones. The two are matched case-insensitively by
#: `_index_of`, exactly as MRST's `strcmpi` does.
_LONG_PHASE_NAMES = {"W": "water", "O": "oil", "G": "gas"}


def _index_of(names, target) -> Optional[int]:
    """Case-insensitive lookup, MRST's `strcmpi`. `None` when absent.

    Absent is a normal outcome, not an error: an equation set that does not
    contain "gas" simply gets no gas source. Raising would break every
    two-phase model.
    """
    key = str(target).lower()
    for i, name in enumerate(names):
        if str(name).lower() == key:
            return i
    return None


def _assert_cell_type(types, index, name) -> None:
    """Source terms are per cell and may only enter a per-cell equation.

    MRST asserts this. Adding a cell-indexed source to a well equation would
    index rows that mean something else entirely, and the result is a solvable
    system answering a different question.
    """
    if index < len(types) and str(types[index]).lower() != "cell":
        raise ValueError(
            f"cannot add source terms to equation {name!r} of type "
            f"{types[index]!r}; source terms are per cell")


def _force_field(obj, name, default=None):
    """Read a field from a struct-like force, dict or namespace."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["ReservoirModel"]
