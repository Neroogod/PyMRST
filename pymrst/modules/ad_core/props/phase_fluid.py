"""Phase-fluid adapter (MRST AD-core phase-fluid layout)."""

from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from pymrst.utils.adi import value


class PhaseFluid:
    """Uniform per-phase view of a PyMRST fluid for state functions."""

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
    """Single-phase water fluid: b(p), constant viscosity, kr = 1."""
    return PhaseFluid(
        phases=["water"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [b(p)],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [mu],
        kr=lambda s, **kw: [
            np.ones_like(value(s[0])) if not hasattr(s[0], "jac") else s[0] * 0.0 + 1.0
        ],
        rho_s=np.array([rho]),
    )


def simple_phase_fluid(fluid) -> PhaseFluid:
    """Adapt a two-phase ``ADFluid`` to the state-function graph."""
    pc_ow = getattr(fluid, "pc_ow", None)
    pc = None if pc_ow is None else (lambda s: [-pc_ow(s[0]), 0.0])
    return PhaseFluid(
        phases=["water", "oil"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [fluid.b_w(p), fluid.b_o(p)],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [fluid.mu[0], fluid.mu[1]],
        kr=lambda s, **kw: [fluid.kr_w(s[0]), fluid.kr_o(s[1])],
        rho_s=fluid.rhoS,
        pc=pc,
    )


def blackoil_phase_fluid(fluid, vapoil: bool = False) -> PhaseFluid:
    """Adapt a black-oil fluid to the state-function graph."""
    return PhaseFluid(
        phases=["water", "oil", "gas"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.b_w(p),
            fluid.b_o(p, rs, sat),
            fluid.b_g(p, rv, sat_g),
        ],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [
            fluid.mu_w(p),
            fluid.mu_o(p, rs, sat),
            fluid.mu_g(p, rv, sat_g),
        ],
        kr=lambda s, **kw: [
            fluid.kr_w(s[0]),
            fluid.kr_o(s[0], s[1], s[2]),
            fluid.kr_g(s[2]),
        ],
        rho_s=fluid.rhoS,
        pv_mult=fluid.pv_mult,
        rs_sat=fluid.rs_sat,
        disgas=True,
        rv_sat=fluid.rv_sat,
        vapoil=vapoil,
    )


__all__ = [
    "PhaseFluid",
    "blackoil_phase_fluid",
    "simple_phase_fluid",
    "water_phase_fluid",
]
