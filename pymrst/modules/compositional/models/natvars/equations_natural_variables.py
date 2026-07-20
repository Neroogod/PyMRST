"""Port surface for MRST ``natvars/equationsNaturalVariables.m``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pymrst.utils.adi import value


@dataclass
class NaturalVariablesProblem:
    """MRST-like natural-variable equation container."""

    equations: list[Any]
    equationNames: list[str]
    types: list[str]
    state: Any
    primaryVariables: list[str]
    reduceToPressure: bool = False
    iterationNo: int = -1
    A: Any = None
    b: np.ndarray | None = None
    assembled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def equation_names(self) -> list[str]:
        return self.equationNames

    @property
    def primary_variables(self) -> list[str]:
        return self.primaryVariables

    def assembleSystem(self):
        """Assemble combined residual and Jacobian if AD equations are present."""
        combined = self.metadata["model"]._combine_equations_for_backend(*self.equations)
        self.b = np.asarray(value(combined), dtype=float).reshape(-1)
        jac = getattr(combined, "jac", None)
        self.A = None if not jac else jac[0]
        self.assembled = True
        return self

    assemble_system = assembleSystem

    def norm(self, ord=np.inf) -> float:
        vals = [np.asarray(value(eq), dtype=float).reshape(-1) for eq in self.equations]
        if not vals:
            return 0.0
        merged = np.concatenate(vals)
        return float(np.linalg.norm(merged, ord=ord))


def _iter_options(args, kwargs):
    opts = dict(kwargs)
    if len(args) == 1 and isinstance(args[0], dict):
        opts.update(args[0])
    elif args:
        if len(args) % 2:
            raise TypeError("positional options must be name/value pairs or a single dict")
        opts.update({str(k): v for k, v in zip(args[0::2], args[1::2])})
    aliases = {
        "Verbose": "verbose",
        "reverseMode": "reverse_mode",
        "resOnly": "res_only",
        "staticWells": "static_wells",
        "propsPressure": "props_pressure",
        "reduceToPressure": "reduce_to_pressure",
        "iteration": "iteration",
    }
    return {aliases.get(str(k), str(k)): v for k, v in opts.items()}


def equations_natural_variables(state0, state, model, dt, drivingForces=None, *args, **kwargs):
    """Build PyMRST natural-variable equations with MRST's call signature."""
    opts = {
        "verbose": False,
        "reverse_mode": False,
        "res_only": False,
        "static_wells": False,
        "props_pressure": None,
        "reduce_to_pressure": False,
        "iteration": -1,
    }
    opts.update(_iter_options(args, kwargs))
    del opts["verbose"], opts["reverse_mode"], opts["props_pressure"]

    st = state
    if st.flag is None:
        st.flag = model._flag_from_flash(st.pressure, st.z)
    _, _, two_phase = model.getFlag(st)
    ncomp = model.mix.n
    nw = len(model.wells)
    primary_values = [
        st.pressure,
        *[st.x[:, i] for i in range(ncomp - 1)],
    ]
    primary_names = [
        "pressure",
        *[f"v_{name}" for name in model.mix.names[:-1]],
    ]
    if model.water:
        primary_values.append(st.sW)
        primary_names.append("sw")
    primary_values.extend([
        st.sO[two_phase],
        *[st.y[two_phase, i] for i in range(ncomp - 1)],
        st.sG[two_phase],
    ])
    primary_names.extend([
        "sl",
        *[f"w_{name}" for name in model.mix.names[:-1]],
        "sv",
    ])
    if nw and not opts["static_wells"]:
        if model.water:
            primary_values.append(st.qWs)
            primary_names.append("qWs")
        primary_values.extend([st.qOs, st.qGs, st.bhp])
        primary_names.extend(["qOs", "qGs", "bhp"])

    prim = tuple(primary_values) if opts["res_only"] else model._init_primary_variables(*primary_values)
    p_ad = prim[0]
    x_prim = list(prim[1:1 + ncomp - 1])
    off = 1 + ncomp - 1
    s_w_ad = None
    if model.water:
        s_w_ad = prim[off]
        off += 1
    s_o_ad = prim[off]
    off += 1
    y_prim = list(prim[off:off + ncomp - 1])
    off += ncomp - 1
    s_g_ad = prim[off]
    off += 1
    qw = None
    if nw and model.water and not opts["static_wells"]:
        qw = prim[off]
        off += 1
    if nw and not opts["static_wells"]:
        qo, qg, bhp = prim[off:off + 3]
    else:
        qo, qg, bhp = st.qOs, st.qGs, st.bhp

    bc = getattr(drivingForces, "bc", None) if drivingForces is not None else None
    src = getattr(drivingForces, "src", None) if drivingForces is not None else None
    eqs, well_eqs, ctrl, fugacity_eqs, closure = model.assemble_natural(
        state0,
        p_ad,
        x_prim,
        s_o_ad,
        y_prim,
        s_g_ad,
        qo,
        qg,
        bhp,
        dt,
        s_w=s_w_ad,
        qw=qw,
        bc=bc,
        src=src,
        flag=st.flag,
    )
    all_eqs = list(eqs)
    names = (["water"] if model.water else []) + list(model.mix.names)
    types = ["cell"] * len(all_eqs)
    if well_eqs:
        all_eqs.extend(well_eqs)
        names.extend([f"well_{i}" for i in range(len(well_eqs))])
        types.extend(["well"] * len(well_eqs))
    if ctrl is not None:
        all_eqs.append(ctrl)
        names.append("control")
        types.append("well")
    all_eqs.extend(fugacity_eqs)
    names.extend([f"fugacity_{name}" for name in model.mix.names])
    types.extend(["cell"] * len(fugacity_eqs))
    all_eqs.append(closure)
    names.append("closure")
    types.append("cell")

    problem = NaturalVariablesProblem(
        all_eqs,
        names,
        types,
        st,
        primary_names,
        reduceToPressure=bool(opts["reduce_to_pressure"]),
        iterationNo=int(opts["iteration"]),
        metadata={
            "model": model,
            "two_phase": two_phase,
            "res_only": bool(opts["res_only"]),
            "static_wells": bool(opts["static_wells"]),
        },
    )
    return problem, st


def equationsNaturalVariables(state0, state, model, dt, drivingForces=None, *args, **kwargs):
    """MRST camelCase alias for :func:`equations_natural_variables`."""
    return equations_natural_variables(state0, state, model, dt, drivingForces, *args, **kwargs)


__all__ = ["NaturalVariablesProblem", "equations_natural_variables", "equationsNaturalVariables"]
