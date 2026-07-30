"""MRST `FlowPropertyFunctions` saturation and component-mass properties."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
    StateFunctionGrouping,
)


class Saturations(StateFunction):
    """Phase saturations, with live-oil hydrocarbon variables resolved."""

    def __init__(self, model=None):
        super().__init__(model, label="s_\\alpha")
        self.single_phase = model is not None and model.fluid.n_phases == 1
        if not self.single_phase:
            self.depends_on("sW", STATE)
        if model is not None and model.fluid.disgas:
            self.depends_on(["sG", "status"], STATE)

    def evaluate_on_domain(self, model, state):
        if self.single_phase:
            return [np.ones(model.G.cells.num)]
        sw = state["sW"]
        if not model.fluid.disgas:
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

    def evaluate_on_domain(self, model, state):
        kr = self.get_evaluated(state, "RelativePermeability")
        mu = self.get_evaluated_external(model, state, "PVTPropertyFunctions",
                                         "Viscosity")
        return [k / m for k, m in zip(kr, mu)]


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

    def evaluate_on_domain(self, model, state):
        per_phase = self.get_evaluated(state, "ComponentPhaseMass")
        return [_sum_over_phases(m) for m in per_phase]


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
        super().__init__("FlowProps")
        self.set_state_function("Saturations", Saturations(model))
        relative_permeability = RelativePermeability(model)
        regions = getattr(model.rock, "regions", None)
        if isinstance(regions, dict):
            regions = regions.get("saturation")
        elif regions is not None:
            regions = getattr(regions, "saturation", None)
        if regions is not None:
            relative_permeability.set_regions(regions)
        self.set_state_function("RelativePermeability", relative_permeability)
        self.set_state_function("Mobility", Mobility(model))
        if model.fluid.has_pc:
            self.set_state_function("CapillaryPressure",
                                    CapillaryPressure(model))
        self.set_state_function("ComponentPhaseMass", ComponentPhaseMass(model))
        self.set_state_function("ComponentTotalMass", ComponentTotalMass(model))


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
