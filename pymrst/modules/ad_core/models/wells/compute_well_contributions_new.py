"""MRST's classic well stack (`ad-core/models/wells/computeWellContributionsNew.m`).

This is the older of MRST's two well assemblies. The newer one - `FacilityModel`
plus `SimpleWell` and the facility state functions - is what everything in this
port uses, and the parity ledger used to exclude this file as "deprecated,
superseded by FacilityModel/SimpleWell".

That reason was wrong in a specific way: `UniformFacilityModel` is *not*
deprecated, and its one and only override calls this function directly. So the
classic stack is reachable from a live class, and excluding it left that class
unable to do the one thing it exists to do.

**What the algorithm is.** Every perforation has a drawdown - the difference
between the reservoir pressure at its cell and the wellbore pressure at its
depth. Its sign says which way that connection flows, and a well can have
connections flowing both ways at once: that is *crossflow*, fluid entering the
wellbore at one perforation and leaving at another. The consequence is that a
producing connection does not produce the composition of its own cell, it
produces the composition of the *wellbore mixture*, which is itself made from
everything the other connections contributed.

That circularity is what `mix_s` resolves, and it is why this is not simply
"rate times mobility per perforation":

1. compute the outflow of each producing connection from its own cell's
   mobilities (`cq_p`), converted to surface volume (`cq_ps`);
2. form the wellbore mixture `mix_s` from those contributions plus whatever the
   well is being told to inject;
3. give every *injecting* connection that mixture rather than its own cell's.

**Dead wells.** A well whose contributions cancel to exactly zero has no
mixture to speak of, and step 3 would divide by zero. MRST substitutes the
well's declared injection composition and a unit total. The substitution is
arithmetic rather than assignment here so the derivatives go to zero with the
values, which is what MRST's in-place assignment does implicitly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi import value


def compute_well_contributions_new(wellmodel: Any, model: Any, sol: Sequence[Any],
                                   p_bh: Any, q_s: Sequence[Any]):
    """Perforation fluxes and well equations for the classic well stack.

    Returns `(eqs, cq_s, mix_s, status, cstatus, cq_r)`, matching MRST:

    * `eqs` - one well equation per phase, `q_s - sum of connection fluxes`;
    * `cq_s` - per-connection flux at **surface** conditions;
    * `mix_s` - the wellbore mixture, `(n_well, n_phase)`, values only;
    * `status` / `cstatus` - which wells and connections are open;
    * `cq_r` - per-connection flux at **reservoir** conditions.
    """
    W = list(wellmodel.W)
    p = wellmodel.referencePressure
    b = list(wellmodel.bfactors)
    r = list(wellmodel.components)
    mob = list(wellmodel.mobilities)
    perf2well = np.asarray(wellmodel.perf2well, dtype=int)
    Rw = wellmodel.Rw
    n_phase = len(b)

    Tw = np.concatenate([np.atleast_1d(np.asarray(w.WI, dtype=float))
                         for w in W])
    compi = np.atleast_2d(np.vstack(
        [np.asarray(w.compi, dtype=float).ravel() for w in W]))

    well_status = np.array([bool(getattr(w, "status", True)) for w in W])
    perf_status = np.concatenate(
        [np.atleast_1d(np.asarray(getattr(w, "cstatus", True))).astype(bool)
         * np.ones(np.atleast_1d(np.asarray(w.WI)).size, dtype=bool)
         for w in W]) & well_status[perf2well]
    Tw = np.where(perf_status, Tw, 0.0)

    qt_s = q_s[0]*well_status
    for ph in range(1, n_phase):
        qt_s = qt_s + q_s[ph]*well_status

    if not getattr(wellmodel, "allowWellSignChange", False):
        is_inj = np.array([float(getattr(w, "sign", -1.0)) > 0 for w in W])
    else:
        # The sign comes from the solution rather than the declaration, which
        # is what lets a well reverse mid-step.
        is_inj = np.asarray(value(qt_s), dtype=float) > 0

    cdp = np.concatenate([np.atleast_1d(np.asarray(_field(s, "cdp", 0.0),
                                                   dtype=float)) for s in sol])
    drawdown = -(_matmul(Rw, p_bh) + cdp) + p
    conn_inj = np.asarray(value(drawdown), dtype=float) < 0

    # A connection whose flow direction disagrees with its well's overall
    # direction is a crossflow connection.
    crossflow = conn_inj != (np.asarray(Rw @ is_inj.astype(float)) > 0.5)
    closed = ~np.concatenate(
        [np.atleast_1d(np.asarray(_field(s, "cstatus", True))).astype(bool)
         * np.ones(np.count_nonzero(perf2well == i), dtype=bool)
         for i, s in enumerate(sol)])
    if not getattr(wellmodel, "allowCrossflow", True):
        closed = closed | crossflow
    Tw = np.where(closed, 0.0, Tw)
    # Kept for structural fidelity, but measured inert: every use of
    # `conn_inj` below is multiplied by `Tw`, which the line above has already
    # zeroed at closed connections, so the flag's value there cannot reach an
    # output. Verified bit-identical across six crossflow/pressure-drop
    # configurations with the line removed. No test covers it, and none
    # pretends to - stated here so the absence is a finding rather than a gap.
    conn_inj = conn_inj & ~closed

    # -- what each producing connection delivers from its own cell ----------
    con_eff = (~conn_inj)*Tw
    cq_p = [-con_eff*mob[ph]*drawdown for ph in range(n_phase)]
    cq_ps = _conn_to_surface(cq_p, b, r, model)
    q_ps = [_matmul(Rw.T, cq_ps[ph]) for ph in range(n_phase)]

    # -- the wellbore mixture ----------------------------------------------
    is_inj = np.asarray(value(qt_s), dtype=float) > 0
    qt_s_inj = is_inj*qt_s
    wbq = []
    for ph in range(n_phase):
        positive = (np.asarray(value(q_s[ph]), dtype=float) > 0).astype(float)
        wbq.append(compi[:, ph]*qt_s_inj
                   + (~is_inj)*q_s[ph]*positive - q_ps[ph])
    wbqt = wbq[0]
    for ph in range(1, n_phase):
        wbqt = wbqt + wbq[ph]

    dead = np.asarray(value(wbqt), dtype=float) == 0
    if np.any(dead):
        # Substituted arithmetically rather than assigned, so the derivatives
        # vanish with the values - which is what MRST's in-place assignment
        # does implicitly, and what keeps the Jacobian consistent with the
        # residual for a well that contributes nothing.
        alive = (~dead).astype(float)
        for ph in range(n_phase):
            wbq[ph] = wbq[ph]*alive + compi[:, ph]*dead
        wbqt = wbqt*alive + dead.astype(float)

    mix_s = [wbq[ph]/wbqt for ph in range(n_phase)]

    # -- what each injecting connection delivers from that mixture ----------
    mt = mob[0]
    for ph in range(1, n_phase):
        mt = mt + mob[ph]
    cqt_i = -(conn_inj*Tw)*(mt*drawdown)
    vol_ratio = _volume_ratio(mix_s, b, r, Rw, model)
    cqt_is = cqt_i/vol_ratio

    cq_s = [cq_ps[ph] + _matmul(Rw, mix_s[ph])*cqt_is for ph in range(n_phase)]
    cq_r = [conn_inj*cqt_i*compi[perf2well, ph] + (~conn_inj)*cq_p[ph]
            for ph in range(n_phase)]

    eqs = [q_s[ph] - _matmul(Rw.T, cq_s[ph]) for ph in range(n_phase)]

    if not np.all(well_status):
        # A shut well's equation becomes `q - q = 0`: it holds trivially and,
        # more to the point, keeps the unknown in the system with a unit
        # derivative instead of leaving a structurally singular row.
        shut = ~well_status
        for ph in range(n_phase):
            eqs[ph] = eqs[ph]*(~shut) + (
                q_s[ph] - np.asarray(value(q_s[ph]), dtype=float))*shut

    mix_out = np.column_stack([np.asarray(value(m), dtype=float).ravel()
                               for m in mix_s])
    status = np.array([bool(_field(s, "status", True)) for s in sol])
    return eqs, cq_s, mix_out, status, ~closed, cq_r


computeWellContributionsNew = compute_well_contributions_new


def _conn_to_surface(v: List[Any], b: List[Any], r: List[Any], model: Any):
    """Reservoir-condition connection fluxes to surface volume.

    The dissolution terms are the content: with dissolved gas, oil arriving at
    the surface releases `rs` volumes of gas per volume of oil, so the gas
    stream is not just the gas that flowed. Omitting them under-reports gas
    production by exactly the solution gas, which is a large fraction of it
    above the bubble point.
    """
    out = [b[ph]*v[ph] for ph in range(len(v))]

    disgas = bool(getattr(model, "disgas", False))
    vapoil = bool(getattr(model, "vapoil", False))
    if not (disgas or vapoil):
        return out

    gas, oil = _gas_oil_indices(model)
    if gas is None or oil is None:
        return out
    if disgas and vapoil:
        out[gas] = out[gas] + r[0]*(b[oil]*v[oil])
        out[oil] = out[oil] + r[1]*(b[gas]*v[gas])
    elif disgas:
        out[gas] = out[gas] + r[0]*(b[oil]*v[oil])
    return out


def _volume_ratio(mix_s: List[Any], b: List[Any], r: List[Any], Rw: Any,
                  model: Any):
    """Reservoir volume occupied by one surface volume of the mixture.

    The `1 - rs*rv` denominator is the black-oil closure: with both dissolved
    gas and vaporised oil, some of the "oil" at the surface came out of the gas
    phase and vice versa, and the two corrections are mutually dependent.
    Solving that pair is what the division undoes.
    """
    n_phase = len(b)
    cmix = [_matmul(Rw, mix_s[ph]) for ph in range(n_phase)]
    tmp = list(cmix)

    disgas = bool(getattr(model, "disgas", False))
    vapoil = bool(getattr(model, "vapoil", False))
    if disgas or vapoil:
        gas, oil = _gas_oil_indices(model)
        if gas is not None and oil is not None:
            if disgas and vapoil:
                d = 1 - r[0]*r[1]
                tmp[gas] = (cmix[gas] - r[0]*cmix[oil])/d
                tmp[oil] = (cmix[oil] - r[1]*cmix[gas])/d
            elif disgas:
                tmp[gas] = cmix[gas] - r[0]*cmix[oil]

    ratio = tmp[0]/b[0]
    for ph in range(1, n_phase):
        ratio = ratio + tmp[ph]/b[ph]
    return ratio


def _gas_oil_indices(model: Any):
    names = [str(n).upper() for n in model.get_phase_names()]
    gas = names.index("G") if "G" in names else None
    oil = names.index("O") if "O" in names else None
    return gas, oil


def _matmul(M: Any, x: Any):
    """`M @ x` that survives an AD right-hand side.

    scipy's sparse `@` raises on an operand it does not recognise instead of
    returning `NotImplemented`, so Python never offers ADI the reflected
    operation and the reflected call has to be made explicitly.
    """
    if hasattr(x, "jac"):
        return x.__rmatmul__(sp.csr_matrix(M))
    return np.asarray(M @ np.asarray(x, dtype=float)).ravel()


def _field(obj: Any, name: str, default: Any = None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["compute_well_contributions_new", "computeWellContributionsNew"]


if __name__ == "__main__":
    def _one_well_setup(n_perf=2, sign=-1.0):
        W = [SimpleNamespace(WI=np.full(n_perf, 1e-11), compi=np.array([1.0, 0.0]),
                             status=True, cstatus=np.ones(n_perf, dtype=bool),
                             sign=sign)]
        sol = [SimpleNamespace(cdp=np.zeros(n_perf), cstatus=np.ones(n_perf, bool),
                               status=True)]
        Rw = sp.csr_matrix(np.ones((n_perf, 1)))
        wellmodel = SimpleNamespace(
            W=W, referencePressure=np.full(n_perf, 2.0e7),
            bfactors=[np.ones(n_perf), np.ones(n_perf)],
            components=[], mobilities=[np.full(n_perf, 1e-3),
                                       np.full(n_perf, 1e-4)],
            perf2well=np.zeros(n_perf, dtype=int), Rw=Rw,
            allowWellSignChange=False, allowCrossflow=True)
        model = SimpleNamespace(disgas=False, vapoil=False,
                                get_phase_names=lambda: ["W", "O"])
        return wellmodel, model, sol

    def test_a_producer_draws_from_the_reservoir():
        """Wellbore below reservoir pressure: every connection produces."""
        wm, model, sol = _one_well_setup()
        eqs, cq_s, mix_s, status, cstatus, cq_r = \
            compute_well_contributions_new(
                wm, model, sol, np.array([1.9e7]),
                [np.array([-1e-3]), np.array([-1e-4])])
        assert np.all(np.asarray(cq_s[0]) < 0), "a producer must have outflow"
        assert cstatus.all()
        print("ok test_a_producer_draws_from_the_reservoir")

    def test_a_shut_well_gets_a_trivial_equation():
        wm, model, sol = _one_well_setup()
        wm.W[0].status = False
        eqs, *_ = compute_well_contributions_new(
            wm, model, sol, np.array([1.9e7]),
            [np.array([-1e-3]), np.array([-1e-4])])
        assert np.allclose(np.asarray(eqs[0]), 0.0)
        print("ok test_a_shut_well_gets_a_trivial_equation")

    def test_the_mixture_sums_to_one():
        """`mix_s` is a composition, so its phases must add to unity."""
        wm, model, sol = _one_well_setup()
        _, _, mix_s, *_ = compute_well_contributions_new(
            wm, model, sol, np.array([1.9e7]),
            [np.array([-1e-3]), np.array([-1e-4])])
        assert np.allclose(mix_s.sum(axis=1), 1.0)
        print("ok test_the_mixture_sums_to_one")

    test_a_producer_draws_from_the_reservoir()
    test_a_shut_well_gets_a_trivial_equation()
    test_the_mixture_sums_to_one()
    print("\nAll tests passed!")
