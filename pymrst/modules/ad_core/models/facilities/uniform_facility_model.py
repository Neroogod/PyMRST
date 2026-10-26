"""MRST ``ad-core/models/facilities/UniformFacilityModel.m``.

This class was a wrapper: a name, a base, and a docstring claiming a
"simplified facility model", inheriting everything and overriding nothing. It
therefore behaved exactly like `FacilityModel`, so asking for it changed
nothing while reading as a choice.

MRST's version overrides one method, `getWellContributions`, and the override
is what the class *is*. It differs from `FacilityModel` in three ways:

* it **refuses components** - MRST asserts `ncomp == 0` - and refuses extra
  well equations. That restriction is the "uniform" in the name: every well
  carries the same phase set and nothing else;
* it assembles through the **classic well stack**,
  `compute_well_contributions_new` and `setup_well_control_equations`, rather
  than through the facility state functions;
* it updates connection pressure drop and well limits per well first.

Both classic-stack functions were previously excluded from the parity ledger
as "deprecated, superseded by FacilityModel/SimpleWell". That reason was wrong
in a specific way, and the wrongness is why this file used to raise:
`UniformFacilityModel` is not deprecated and calls them directly, so the
classic stack is reachable from a live class. They are ported now, and this is
the class that uses them.

**Why keep two assemblies at all.** They are not redundant. The state-function
path computes each well's contributions through the property graph, which is
what lets components, thermal terms and per-well extensions participate. The
classic path computes them in one pass from mobilities and b-factors, which is
faster and easier to reason about but cannot express any of that. MRST keeps
both and so does this port; the restriction is enforced rather than assumed,
so a caller who attaches components to a uniform facility is told, not quietly
given a model that ignores them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.wells.compute_well_contributions_new import (
    compute_well_contributions_new,
)
from pymrst.modules.ad_core.models.wells.setup_well_control_equations import (
    setup_well_control_equations,
)
from pymrst.modules.ad_core.utils.get_perforation_to_well_mapping import (
    get_perforation_to_well_mapping,
)
from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)
from pymrst.utils.adi import value


class UniformFacilityModel(FacilityModel):
    """MRST's uniform facility model: the classic single-pass well assembly."""

    def get_well_contributions(self, wellSol0: Any = None, wellSol: Any = None,
                               wellvars: Any = None, wellMap: Any = None,
                               p: Any = None, mob: Any = None, rho: Any = None,
                               dissolved: Any = None, comp: Any = None,
                               dt: float = None, iteration: Any = None,
                               **kwargs: Any):
        """Assemble well equations through the classic stack.

        Returns `(sources, well_system, wellSol)` - the same triple as
        `FacilityModel.get_well_contributions`, so the two are interchangeable
        at the call site even though they compute it differently.
        """
        self._refuse_extensions()

        active = np.atleast_1d(self.get_indices_of_active_wells(wellSol))
        wells = [getattr(self.WellModels[int(i)], "W", self.WellModels[int(i)])
                 for i in active]
        sols = [wellSol[int(i)] for i in active] if wellSol is not None else []

        cells = self.get_well_cells(active)
        model = self.ReservoirModel
        n_phase = len(mob)

        # Everything the classic stack reads is *at the perforated cells*.
        p_w = p[cells] if p is not None else None
        mob_w = [m[cells] for m in mob]
        rho_w = [r[cells] for r in rho]
        dissolved_w = _subset_dissolved(dissolved, cells)

        rho_s = np.asarray(model.get_surface_densities(), dtype=float).ravel()
        b_w = phase_densities_to_bfactor(rho_w, rho_s, dissolved_w)

        perf2well, Rw = get_perforation_to_well_mapping(wells)
        first = self.WellModels[int(active[0])] if active.size else None
        wellmodel = SimpleNamespace(
            W=wells, referencePressure=p_w, bfactors=b_w,
            components=_black_oil_ratios(model, dissolved_w),
            mobilities=mob_w, perf2well=perf2well, Rw=Rw,
            allowWellSignChange=bool(getattr(first, "allow_sign_change",
                                             getattr(first, "allowSignChange",
                                                     False))),
            allowCrossflow=bool(getattr(first, "allow_crossflow",
                                        getattr(first, "allowCrossflow",
                                                True))))

        bhp, q_well = _split_well_variables(wellvars, wellMap, n_phase)

        eqs, cq_s, mix_s, status, cstatus, cq_r = \
            compute_well_contributions_new(wellmodel, model, sols, bhp, q_well)
        control = setup_well_control_equations(sols, bhp, q_well, status,
                                               mix_s, model)

        # The classic stack works in surface volume; the sources the reservoir
        # equations want are mass, hence the density weighting here rather
        # than inside the stack - the stack is shared with callers that want
        # surface volume.
        phase_mass = [np.asarray(value(cq_s[ph]), dtype=float)*rho_s[ph]
                      if not hasattr(cq_s[ph], "jac") else cq_s[ph]*rho_s[ph]
                      for ph in range(n_phase)]

        phases = self._phase_names()[:n_phase]
        well_system = SimpleNamespace(
            wellEquations=list(eqs), well_equations=list(eqs),
            names=[f"well{ph.upper()}" for ph in phases],
            types=["perf"]*n_phase,
            controlEquation=control, control_equation=control)
        sources = SimpleNamespace(
            phaseMass=phase_mass, phase_mass=phase_mass,
            phaseVolume=list(cq_r), phase_volume=list(cq_r),
            components=[], sourceCells=cells, source_cells=cells)
        return sources, well_system, wellSol

    getWellContributions = get_well_contributions

    def _refuse_extensions(self) -> None:
        """MRST's two asserts, kept as errors.

        Silently falling back to the general assembly would be the wrapper
        behaviour this class replaces: the caller asked for the uniform stack
        and would have received the state-function one, with no indication the
        restriction was ignored - or worse, received the uniform one with the
        components dropped.
        """
        reservoir = getattr(self, "ReservoirModel", None)
        components = list(reservoir.get_component_names()) \
            if reservoir is not None else []
        if components:
            raise NotImplementedError(
                f"UniformFacilityModel does not support components; the model "
                f"declares {components}. The classic well stack has no "
                f"component transport - use FacilityModel, whose "
                f"state-function assembly does.")
        extra = list(getattr(self, "addedEquationNames", []) or [])
        if extra:
            raise NotImplementedError(
                f"UniformFacilityModel does not support extra well equations; "
                f"{extra} are attached. Use FacilityModel.")


def _split_well_variables(wellvars, well_map, n_phase):
    """Separate the BHP and rate unknowns from the packed well variables."""
    if well_map is None:
        variables = list(wellvars or [])
        return variables[0], variables[1:1 + n_phase]

    is_bhp = np.atleast_1d(np.asarray(
        _field(well_map, "isBHP", _field(well_map, "is_bhp", [])), dtype=bool))
    is_rate = np.atleast_1d(np.asarray(
        _field(well_map, "isRate", _field(well_map, "is_rate", [])), dtype=bool))
    variables = list(wellvars or [])
    bhp = next((v for v, flag in zip(variables, is_bhp) if flag), None)
    rates = [v for v, flag in zip(variables, is_rate) if flag]
    return bhp, rates


def _subset_dissolved(dissolved, cells):
    """Restrict the dissolution table to the perforated cells."""
    if not dissolved:
        return dissolved
    out = []
    for row in dissolved:
        out.append([None if entry is None else entry[cells] for entry in row])
    return out


def _black_oil_ratios(model, dissolved_w):
    """`[rs, rv]` for the classic stack, or `[]` for a dead-oil fluid.

    MRST indexes the dissolution table by phase position; this reads the same
    two entries by name, so a two-phase model - where the positions differ -
    reaches the right ones instead of the wrong ones.
    """
    disgas = bool(getattr(model, "disgas", False))
    vapoil = bool(getattr(model, "vapoil", False))
    if not (disgas or vapoil) or not dissolved_w:
        return []

    names = [str(n).upper() for n in model.get_phase_names()]
    if "G" not in names or "O" not in names:
        return []
    gas, oil = names.index("G"), names.index("O")

    def entry(i, j):
        try:
            found = dissolved_w[i][j]
        except (IndexError, TypeError):
            return 0.0
        return 0.0 if found is None else found

    return [entry(gas, oil), entry(oil, gas)]


def _field(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["UniformFacilityModel"]
