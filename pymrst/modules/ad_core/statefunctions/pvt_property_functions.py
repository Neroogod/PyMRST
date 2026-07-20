"""MRST `PVTPropertyFunctions` and pressure-driven fluid properties."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
    StateFunctionGrouping,
)
from pymrst.utils.adi import value


class PhaseFluid:
    """Uniform per-phase view of a PyMRST fluid, for the state-function graph."""

    def __init__(
        self,
        phases: Sequence[str],
        b: Callable[..., List[Any]],
        mu: Callable[..., List[Any]],
        kr: Callable[..., List[Any]],
        rho_s: np.ndarray,
        pv_mult: Optional[Callable[[Any], Any]] = None,
        rs_sat: Optional[Callable[[Any], Any]] = None,
        disgas: bool = False,
        pc: Optional[Callable[[List[Any]], List[Any]]] = None,
        rv_sat: Optional[Callable[[Any], Any]] = None,
        vapoil: bool = False,
    ):
        self.phases = list(phases)
        self.b = b
        self.mu = mu
        self.kr = kr
        self.rho_s = np.asarray(rho_s, dtype=float)
        self.pv_mult = pv_mult or (lambda p: 1.0)
        self.rs_sat = rs_sat
        self.disgas = disgas
        self.rv_sat = rv_sat
        self.vapoil = vapoil
        self.pc = pc

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    @property
    def has_pc(self) -> bool:
        return self.pc is not None


def water_phase_fluid(b, mu: float, rho: float) -> PhaseFluid:
    """A single-phase water fluid: b(p), constant viscosity, kr = 1."""
    return PhaseFluid(
        phases=["water"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [b(p)],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [mu],
        kr=lambda s, **kw: [
            np.ones_like(value(s[0])) if not hasattr(s[0], "jac")
            else s[0] * 0.0 + 1.0
        ],
        rho_s=np.array([rho]),
    )


def simple_phase_fluid(fluid) -> PhaseFluid:
    """Adapt the two-phase `ADFluid` of `ad_props.simple_adi_fluid`."""
    pc_ow = getattr(fluid, "pc_ow", None)
    pc = None if pc_ow is None else (lambda s: [-pc_ow(s[0]), 0.0])
    return PhaseFluid(
        phases=["water", "oil"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.b_w(p), fluid.b_o(p)
        ],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.mu[0], fluid.mu[1]
        ],
        kr=lambda s, **kw: [fluid.kr_w(s[0]), fluid.kr_o(s[1])],
        rho_s=fluid.rhoS,
        pc=pc,
    )


def blackoil_phase_fluid(fluid, vapoil: bool = False) -> PhaseFluid:
    """Adapt the three-phase live-oil / wet-gas `BlackOilFluid`."""
    pc_ow = getattr(fluid, "pc_ow", None)
    pc_og = getattr(fluid, "pc_og", None)
    pc = None
    if pc_ow is not None or pc_og is not None:
        def pc(s):
            pcow = 0.0 if pc_ow is None else pc_ow(s[0])
            pcog = 0.0 if pc_og is None else pc_og(s[2])
            return [-pcow, 0.0, pcog]

    return PhaseFluid(
        phases=["water", "oil", "gas"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.b_w(p), fluid.b_o(p, rs, sat), fluid.b_g(p, rv, sat_g)
        ],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.mu_w(p), fluid.mu_o(p, rs, sat), fluid.mu_g(p, rv, sat_g)
        ],
        kr=lambda s, **kw: [
            fluid.kr_w(s[0]), fluid.kr_o(s[0], s[1], s[2]), fluid.kr_g(s[2])
        ],
        rho_s=fluid.rhoS,
        pv_mult=fluid.pv_mult,
        rs_sat=fluid.rs_sat,
        disgas=True,
        rv_sat=fluid.rv_sat,
        vapoil=vapoil,
        pc=pc,
    )


class _PVTFunction(StateFunction):
    """Base for pressure-driven fluid properties."""

    def __init__(self, model=None, label=None):
        super().__init__(model, label)
        self.depends_on("pressure", STATE)
        if model is not None and model.fluid.disgas:
            self.depends_on(["rs", "status"], STATE)
        if model is not None and model.fluid.vapoil:
            self.depends_on(["rv", "status"], STATE)

    @staticmethod
    def _pvt_args(model, state):
        p = state["pressure"]
        f = model.fluid
        if not (f.disgas or f.vapoil):
            return p, None, None, None, None
        status = state["status"]
        gas_present = status >= 2
        oil_present = (status % 2) == 1
        rs = state["rs"] if f.disgas else None
        rv = state["rv"] if f.vapoil else None
        return p, rs, gas_present, rv, oil_present


class PoreVolume(_PVTFunction):
    """Pore volume, pressure-dependent through the rock multiplier."""

    def evaluate_on_domain(self, model, state):
        p = state["pressure"]
        return model.pv_ref * model.fluid.pv_mult(p)


class _PhasePressureFunction(_PVTFunction):
    """A PVT property evaluated at each phase's own pressure."""

    def __init__(self, model=None, label=None):
        super().__init__(model, label)
        self.per_phase_p = model is not None and model.fluid.has_pc
        if self.per_phase_p:
            self.depends_on("PhasePressures")

    def _phase_pressures(self, model, state):
        if not self.per_phase_p:
            return [state["pressure"]] * model.n_phases
        return self.get_evaluated(state, "PhasePressures")


class ShrinkageFactors(_PhasePressureFunction):
    """b_alpha: reservoir volume -> surface volume."""

    def evaluate_on_domain(self, model, state):
        _, rs, sat, rv, sat_g = self._pvt_args(model, state)
        p_phase = self._phase_pressures(model, state)
        if not self.per_phase_p:
            return model.fluid.b(p_phase[0], rs, sat, rv, sat_g)
        return [
            model.fluid.b(p_phase[i], rs, sat, rv, sat_g)[i]
            for i in range(model.n_phases)
        ]


class Viscosity(_PhasePressureFunction):
    """Phase viscosities at each phase pressure."""

    def evaluate_on_domain(self, model, state):
        _, rs, sat, rv, sat_g = self._pvt_args(model, state)
        p_phase = self._phase_pressures(model, state)
        if not self.per_phase_p:
            return model.fluid.mu(p_phase[0], rs, sat, rv, sat_g)
        return [
            model.fluid.mu(p_phase[i], rs, sat, rv, sat_g)[i]
            for i in range(model.n_phases)
        ]


class Density(StateFunction):
    """Phase densities, including dissolved/vaporized hydrocarbon carry."""

    def __init__(self, model=None):
        super().__init__(model)
        self.depends_on("ShrinkageFactors")
        if model is not None and model.fluid.disgas:
            self.depends_on("rs", STATE)
        if model is not None and model.fluid.vapoil:
            self.depends_on("rv", STATE)

    def evaluate_on_domain(self, model, state):
        b = self.get_evaluated(state, "ShrinkageFactors")
        rho_s = model.fluid.rho_s
        f = model.fluid
        if not (f.disgas or f.vapoil):
            return [b[i] * rho_s[i] for i in range(len(b))]

        rho_o = (
            b[1] * (rho_s[1] + state["rs"] * rho_s[2])
            if f.disgas else b[1] * rho_s[1]
        )
        rho_g = (
            b[2] * (rho_s[2] + state["rv"] * rho_s[1])
            if f.vapoil else b[2] * rho_s[2]
        )
        return [b[0] * rho_s[0], rho_o, rho_g]


class PVTPropertyFunctions(StateFunctionGrouping):
    """The pressure-driven fluid properties (MRST PVTPropertyFunctions)."""

    def __init__(self, model):
        from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
            PhasePressures,
        )

        super().__init__("PVTProps")
        self.set_state_function("PoreVolume", PoreVolume(model))
        self.set_state_function("ShrinkageFactors", ShrinkageFactors(model))
        self.set_state_function("Viscosity", Viscosity(model))
        self.set_state_function("Density", Density(model))
        self.set_state_function("PhasePressures", PhasePressures(model))


__all__ = [
    "Density",
    "PVTPropertyFunctions",
    "PhaseFluid",
    "PoreVolume",
    "ShrinkageFactors",
    "Viscosity",
    "blackoil_phase_fluid",
    "simple_phase_fluid",
    "water_phase_fluid",
]
