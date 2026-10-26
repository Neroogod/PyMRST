"""MRST `FlowPropertyFunctions` saturation and component-mass properties."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
    StateFunctionGrouping,
)


class Saturations(StateFunction):
    """Phase saturations, with live-oil hydrocarbon variables resolved.

    How many saturations there are is decided by the **phase count**, not by
    whether the model has dissolution. Those two agree for every case with a
    dissolution table, which is why deciding on `disgas or vapoil` worked for
    so long - but a three-phase *dead-oil, dry-gas* model has neither flag,
    and returning two saturations for it leaves the gas component with no
    phase to live in. `blackoilTutorialMexAcceleration` builds exactly that.
    """

    def __init__(self, model=None):
        super().__init__(model, label="s_\\alpha")
        n_phases = 0 if model is None else int(model.fluid.n_phases)
        self.single_phase = n_phases == 1
        self.three_phase = n_phases > 2
        if not self.single_phase:
            self.depends_on("sW", STATE)
        if self.three_phase:
            self.depends_on("sG", STATE)
        if model is not None and (model.fluid.disgas or model.fluid.vapoil):
            # `status` says which hydrocarbon variable a cell carries, and only
            # a model with dissolution has one.
            self.depends_on("status", STATE)

    def evaluate_on_domain(self, model, state):
        if self.single_phase:
            return [np.ones(model.G.cells.num)]
        sw = state["sW"]
        if not self.three_phase:
            return [sw, 1.0 - sw]
        sg = state["sG"]
        return [sw, 1.0 - sw - sg, sg]


class RelativePermeability(StateFunction):
    """Relative permeability per phase."""

    def __init__(self, model=None):
        super().__init__(model, label="k_{r\\alpha}")
        self.depends_on("Saturations")

    def evaluate_on_domain(self, model, state):
        if self.regions is not None:
            from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis import (
                evaluate_function_cell_subset_reg,
            )

            saturations = self.get_evaluated(state, "Saturations")
            raw = getattr(model, "raw_fluid", None)
            if raw is not None and model.n_phases == 2:
                functions = (raw.kr_w, raw.kr_o)
                if any(isinstance(fn, (list, tuple)) for fn in functions):
                    return [
                        evaluate_function_cell_subset_reg(
                            self, fn, self.regions, saturation
                        )
                        for fn, saturation in zip(functions, saturations)
                    ]
        return model.fluid.kr(self.get_evaluated(state, "Saturations"))


class Mobility(StateFunction):
    """Phase mobility kr/mu."""

    def __init__(self, model=None):
        super().__init__(model, label="\\lambda_\\alpha")
        self.depends_on("RelativePermeability")
        self.depends_on("Viscosity", "PVTPropertyFunctions")
        self.has_multipliers = False

    def enable_multipliers(self, enabled: bool = True) -> "Mobility":
        """Multiply the mobility by a `MobilityMultipliers` property.

        Off by default and opt-in, as in MRST. Enabling it *declares a
        dependency*, which is the part that matters: the grouping evaluates in
        dependency order, so a multiplier that is used without being declared
        is read before it is computed - reading a stale value from the previous
        Newton iteration rather than failing.
        """
        self.has_multipliers = bool(enabled)
        if self.has_multipliers:
            self.depends_on("MobilityMultipliers")
        return self

    enableMultipliers = enable_multipliers

    def evaluate_on_domain(self, model, state):
        kr = self.get_evaluated(state, "RelativePermeability")
        mu = self.get_evaluated_external(model, state, "PVTPropertyFunctions",
                                         "Viscosity")
        mob = [k / m for k, m in zip(kr, mu)]
        if self.has_multipliers:
            multipliers = self.get_evaluated(state, "MobilityMultipliers")
            mob = [m*f for m, f in zip(mob, multipliers)]
        return mob


class CapillaryPressure(StateFunction):
    """Phase pressure offsets from the model reference pressure."""

    def __init__(self, model=None):
        super().__init__(model, label="p_c")
        self.depends_on("Saturations")

    def evaluate_on_domain(self, model, state):
        s = self.get_evaluated(state, "Saturations")
        return model.fluid.pc(s)


class PhasePressures(StateFunction):
    """Each phase's own pressure."""

    def __init__(self, model=None):
        super().__init__(model, label="p_\\alpha")
        self.depends_on("pressure", STATE)
        self.has_pc = model is not None and model.fluid.has_pc
        if self.has_pc:
            self.depends_on("CapillaryPressure", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model, state):
        p = state["pressure"]
        nph = model.n_phases
        if not self.has_pc:
            return [p] * nph
        pc = self.get_evaluated_external(model, state, "FlowPropertyFunctions",
                                         "CapillaryPressure")
        return [p + pc[i] for i in range(nph)]


class ComponentPhaseMass(StateFunction):
    """Amount of each conserved component in each phase."""

    def __init__(self, model=None):
        super().__init__(model, label="m_{i\\alpha}")
        self.depends_on(["Saturations"])
        self.depends_on(["PoreVolume", "ShrinkageFactors"],
                        "PVTPropertyFunctions")
        if model is not None:
            for key in model.component_state_keys():
                self.depends_on(key, STATE)

    def evaluate_on_domain(self, model, state):
        s = self.get_evaluated(state, "Saturations")
        pv = self.get_evaluated_external(model, state, "PVTPropertyFunctions",
                                         "PoreVolume")
        b = self.get_evaluated_external(model, state, "PVTPropertyFunctions",
                                        "ShrinkageFactors")
        rs = state["rs"] if model.fluid.disgas else None
        rv = state["rv"] if model.fluid.vapoil else None
        scale = model._basis_scale()

        out = []
        for c, comp in enumerate(model.components):
            rho = comp.component_density(model, state, b, rs, rv)
            out.append([
                None if rho[ph] is None else scale[c] * pv * s[ph] * rho[ph]
                for ph in range(len(s))
            ])
        return out


class ComponentTotalMass(StateFunction):
    """Total amount of each component per cell."""

    def __init__(self, model=None):
        super().__init__(model, label="m_i")
        self.depends_on("ComponentPhaseMass")
        #: `None` means "leave the derivatives alone", which is the default and
        #: is different from zero - see `ensure_minimum_derivatives`.
        self.minimum_derivatives = None

    def set_minimum_derivatives(self, der=1e-10) -> "ComponentTotalMass":
        """Floor the magnitude of the accumulation derivatives.

        A component that is absent from a cell has zero mass and zero
        derivative, so its row of the Jacobian is empty and the linear solver
        has nothing to move: the component can never appear there, however
        strongly it is driven from a neighbour. A tiny floor keeps the row
        invertible without meaningfully changing a cell where the component
        *is* present, since 1e-10 is far below any real accumulation.

        MRST's default when called with no argument is 1e-10, and it is
        supplied here rather than left to the caller for the same reason: the
        value is a numerical floor, not a physical parameter.
        """
        self.minimum_derivatives = der
        return self

    setMinimumDerivatives = set_minimum_derivatives

    def get_minimum_derivatives(self):
        return self.minimum_derivatives

    getMinimumDerivatives = get_minimum_derivatives

    def ensure_minimum_derivatives(self, mass):
        """Raise any derivative below the floor, in magnitude, keeping its sign.

        `None` returns the input untouched. A floor of *zero* reaches the same
        result by a different route - `abs(d) < 0` is never true, so nothing is
        rewritten - and the only difference is that `None` returns immediately
        while zero walks every Jacobian to change nothing. That makes the two
        indistinguishable by any test of the output, so no test claims
        otherwise; `None` is the default because "no floor" should not cost a
        pass over the Jacobian, not because the numbers differ.

        MRST reaches into the diagonal storage of its `DiagonalAutoDiffBackend`
        here. PyMRST's Jacobians are sparse matrices, so the equivalent is the
        stored entries of the diagonal - a component's own derivative with
        respect to its own cell. Only the stored ones: creating an entry where
        the sparsity pattern has none would change the pattern, and the pattern
        is what the linear solver's symbolic factorisation is built on.
        """
        der = self.minimum_derivatives
        if der is None:
            return mass

        floors = np.atleast_1d(np.asarray(der, dtype=float))
        if floors.size == 1:
            floors = np.repeat(floors, len(mass))

        for c, m in enumerate(mass):
            if not hasattr(m, "jac") or c >= floors.size:
                continue
            floor = float(floors[c])
            for jac in m.jac:
                diagonal = jac.diagonal()
                low = np.abs(diagonal) < floor
                if not np.any(low):
                    continue
                # Preserve the sign; a derivative that is small and negative
                # must not be flipped positive by the floor.
                sign = np.where(diagonal < 0, -1.0, 1.0)
                jac.setdiag(np.where(low, sign*floor, diagonal))
        return mass

    ensureMinimumDerivatives = ensure_minimum_derivatives

    def evaluate_on_domain(self, model, state):
        per_phase = self.get_evaluated(state, "ComponentPhaseMass")
        return self.ensure_minimum_derivatives(
            [_sum_over_phases(m) for m in per_phase])


def region_indicator(model, kind: str):
    """The named region column off `model.rock.regions`, or `None`.

    Shared by every grouping that dispatches on regions, under either
    spelling - `rock.regions` may be a dict or an object.
    """
    regions = getattr(model.rock, "regions", None)
    if regions is None:
        return None
    if isinstance(regions, dict):
        return regions.get(kind)
    return getattr(regions, kind, None)


def _region_or_ones(model, kind: str) -> np.ndarray:
    """MRST's `getRegion*`: the indicator, or region 1 in every cell.

    MRST and PyMRST say "one region" two different ways, and both are kept
    deliberately. MRST returns `ones(nc, 1)`, so a caller always indexes an
    array; PyMRST's constructors use `None` to mean "do not dispatch on
    regions at all", which skips the per-region loop entirely.

    Returning ones here and `None` from `region_indicator` is therefore not a
    duplication - swapping either one changes behaviour. Ones passed to
    `set_regions` would install a one-region dispatch where there had been
    none, which is the same answer computed down a different path, and that
    path is not exercised by any existing test.
    """
    values = region_indicator(model, kind)
    if values is None:
        return np.ones(int(model.G.cells.num), dtype=int)
    return np.asarray(values)


def _sum_over_phases(per_phase):
    """Add a component's phase contributions, skipping absent phases."""
    present = [x for x in per_phase if x is not None]
    assert present, "a component must exist in at least one phase"
    total = present[0]
    for x in present[1:]:
        total = total + x
    return total


class FlowPropertyFunctions(StateFunctionGrouping):
    """Saturation- and rock-driven properties (MRST FlowPropertyFunctions)."""

    def __init__(self, model):
        from pymrst.modules.ad_core.statefunctions.flowprops.component_properties import (
            ComponentMobility, ComponentPhaseDensity,
        )

        super().__init__("FlowProps")
        self.set_state_function("Saturations", Saturations(model))
        relative_permeability = RelativePermeability(model)
        regions = region_indicator(model, "saturation")
        if regions is not None:
            relative_permeability.set_regions(regions)
        self.set_state_function("RelativePermeability", relative_permeability)
        self.set_state_function("Mobility", Mobility(model))
        self.set_state_function("ComponentPhaseDensity", ComponentPhaseDensity(model))
        self.set_state_function("ComponentMobility", ComponentMobility(model))
        if model.fluid.has_pc:
            self.set_state_function("CapillaryPressure",
                                    CapillaryPressure(model))
        self.set_state_function("ComponentPhaseMass", ComponentPhaseMass(model))
        self.set_state_function("ComponentTotalMass", ComponentTotalMass(model))

    # -- region indicators (MRST `getRegionSaturation` / `getRegionSurfactant`)

    def get_region_saturation(self, model) -> np.ndarray:
        """Saturation-region index per cell; region 1 everywhere by default."""
        return _region_or_ones(model, "saturation")

    getRegionSaturation = get_region_saturation

    def get_region_surfactant(self, model) -> np.ndarray:
        """Surfactant-region index per cell; region 1 everywhere by default."""
        return _region_or_ones(model, "surfactant")

    getRegionSurfactant = get_region_surfactant


__all__ = [
    "CapillaryPressure",
    "ComponentPhaseMass",
    "ComponentTotalMass",
    "FlowPropertyFunctions",
    "Mobility",
    "PhasePressures",
    "RelativePermeability",
    "Saturations",
    "_sum_over_phases",
]
