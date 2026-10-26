"""MRST `ad-core/models/wells/setupWellControlEquations.m`.

One closure equation per well, saying "this well is on the control it was
given". Which residual that is depends on the control:

    bhp    ->  p_bh - target
    rate   ->  sum of surface rates - target
    orat   ->  oil rate - target          (likewise wrat, grat)
    lrat   ->  water + oil rate - target
    vrat   ->  sum of surface rates - target

Ported alongside `computeWellContributionsNew` and for the same reason: the
ledger excluded it as deprecated, but `UniformFacilityModel` - which is not
deprecated - calls it directly, so the exclusion made a live class unusable.

Two cases matter more than the table above.

**A rate target the mixture cannot deliver.** A well told to produce oil at a
fixed rate from a wellbore mixture containing no oil has an equation with a
zero derivative in every unknown: nothing the solver changes affects the
residual, and the row is structurally singular. MRST detects the zero mixture
fraction and replaces the equation with "total rate is zero", which is the
physically right answer - the well cannot deliver that phase, so it delivers
nothing - and is solvable. The same substitution covers an explicit zero rate
target, which is how a well is shut in practice.

**A disabled well** keeps its unknown but is pinned to its current pressure.
The residual is scaled by `rate/pressure` magnitudes so that row has the same
weight as the others; unscaled, a pressure residual in pascals sits alongside
rate residuals in m^3/s and the linear solver sees a badly conditioned system
whose worst row is the one that matters least.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.utils.adi import value


def setup_well_control_equations(sol: Sequence[Any], p_bh: Any,
                                 q_s: Sequence[Any], status: np.ndarray,
                                 mix_s: np.ndarray, model: Any):
    """The per-well control closure equation.

    `mix_s` is the wellbore mixture from `compute_well_contributions_new`,
    `(n_well, n_phase)`; `status` says which wells are enabled.
    """
    names = [str(n).upper() for n in model.get_phase_names()]
    iw = names.index("W") if "W" in names else None
    io = names.index("O") if "O" in names else None
    ig = names.index("G") if "G" in names else None

    types = np.array([str(_field(s, "type", "bhp")).lower() for s in sol])
    targets = np.array([float(_field(s, "val", 0.0)) for s in sol])
    bhp = np.array([float(np.ravel(value(_field(s, "bhp", 0.0)))[0])
                    for s in sol])
    status = np.asarray(status, dtype=bool)
    mix_s = np.atleast_2d(np.asarray(mix_s, dtype=float))

    qt_s = q_s[0]
    for ph in range(1, len(q_s)):
        qt_s = qt_s + q_s[ph]

    # A zero rate target on anything but a BHP well means "shut", and is
    # handled by the same substitution as an undeliverable phase.
    to_zero_rate = (targets == 0) & (types != "bhp")

    equation = p_bh*0.0
    residuals = {
        "bhp": lambda: p_bh - targets,
        "rate": lambda: qt_s - targets,
        "vrat": lambda: qt_s - targets,
        "orat": lambda: q_s[io] - targets,
        "wrat": lambda: q_s[iw] - targets,
        "grat": lambda: q_s[ig] - targets,
        "lrat": lambda: q_s[iw] + q_s[io] - targets,
    }
    #: Which mixture fractions a control depends on. An empty tuple means the
    #: control cannot be starved - a BHP or total-rate well always has one.
    depends_on = {"orat": (io,), "wrat": (iw,), "grat": (ig,),
                  "lrat": (iw, io)}

    for mode, build in residuals.items():
        selected = (types == mode) & status
        if not selected.any():
            continue
        phases = depends_on.get(mode, ())
        if any(p is None for p in phases):
            continue                      # the model has no such phase
        equation = _where(selected, build(), equation)

        if phases:
            available = sum(mix_s[:, p] for p in phases)
            to_zero_rate |= selected & (available == 0)

    if to_zero_rate.any():
        equation = _where(to_zero_rate, qt_s, equation)

    disabled = ~status
    if disabled.any():
        # Scaled so this row carries the same weight as the rate rows it sits
        # beside. `sqrt(eps)` and `1` are MRST's floors, and they matter on the
        # first iteration when every rate is still zero.
        rate_scale = max(float(np.mean(np.abs(np.asarray(value(qt_s),
                                                         dtype=float)))),
                         float(np.sqrt(np.finfo(float).eps)))
        bhp_scale = max(float(np.mean(np.abs(bhp))), 1.0)
        equation = _where(disabled,
                          rate_scale*(p_bh - bhp)/bhp_scale, equation)
    return equation


setupWellControlEquations = setup_well_control_equations


def _where(mask: np.ndarray, chosen: Any, otherwise: Any) -> Any:
    """`np.where` that keeps AD derivatives on both branches.

    Written arithmetically rather than by assignment: an ADI object's Jacobian
    rows have to be selected along with its values, and multiplying by a 0/1
    mask does both in one expression. Assigning into `.val` would leave the
    derivative of the replaced branch behind - a residual and a Jacobian
    describing two different equations.
    """
    take = np.asarray(mask, dtype=float)
    return chosen*take + otherwise*(1.0 - take)


def _field(obj: Any, name: str, default: Any = None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["setup_well_control_equations", "setupWellControlEquations"]


if __name__ == "__main__":
    from types import SimpleNamespace

    _MODEL = SimpleNamespace(get_phase_names=lambda: ["W", "O"])

    def _sol(type="bhp", val=2.0e7, bhp=2.0e7, status=True):
        return SimpleNamespace(type=type, val=val, bhp=bhp, status=status)

    def test_a_bhp_well_closes_on_its_pressure():
        eq = setup_well_control_equations(
            [_sol("bhp", 2.0e7)], np.array([1.9e7]),
            [np.array([-1.0]), np.array([-2.0])], np.array([True]),
            np.array([[0.5, 0.5]]), _MODEL)
        assert np.allclose(eq, 1.9e7 - 2.0e7)
        print("ok test_a_bhp_well_closes_on_its_pressure")

    def test_a_rate_well_closes_on_its_total():
        eq = setup_well_control_equations(
            [_sol("rate", -3.0)], np.array([2.0e7]),
            [np.array([-1.0]), np.array([-2.0])], np.array([True]),
            np.array([[0.5, 0.5]]), _MODEL)
        assert np.allclose(eq, 0.0)
        print("ok test_a_rate_well_closes_on_its_total")

    def test_an_undeliverable_phase_becomes_a_zero_rate():
        """No oil in the mixture: the orat equation has no derivative."""
        eq = setup_well_control_equations(
            [_sol("orat", -3.0)], np.array([2.0e7]),
            [np.array([-1.0]), np.array([-2.0])], np.array([True]),
            np.array([[1.0, 0.0]]), _MODEL)
        assert np.allclose(eq, -3.0), "should have become 'total rate is zero'"
        print("ok test_an_undeliverable_phase_becomes_a_zero_rate")

    test_a_bhp_well_closes_on_its_pressure()
    test_a_rate_well_closes_on_its_total()
    test_an_undeliverable_phase_becomes_a_zero_rate()
    print("\nAll tests passed!")
