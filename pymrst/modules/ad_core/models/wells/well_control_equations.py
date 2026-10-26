"""Well control equations (MRST ad-core/models/facilities -> wells).

Ports MRST's `setupWellControlEquationsSingleWell.m` (AC13): the one equation
that closes a well by holding its target. A well has one degree of freedom per
control (its bottom-hole pressure, or a surface rate); the control equation
pins that freedom to the well's target value, and which target depends on the
control type:

- ``bhp``  : ``p_bh - val``
- ``rate`` : ``sum_a q_a - val``           (total surface rate)
- ``wrat`` / ``orat`` / ``grat`` : the single-phase surface rate ``q_a - val``
- ``lrat`` : ``q_water + q_oil - val``     (liquid rate)

with two guards MRST applies: a *shut* well gets a trivial ``p_bh - bhp``
condition, and a rate control whose phase is absent (``mix_s == 0``, a singular
equation) falls back to a zero total-rate condition.

PyMRST computes this today inside `FacilityModel._control_equation`; that path
is MATLAB-validated (Phase 120). This module is the same logic factored onto
MRST's layout as a standalone single-well function, validated to reproduce the
procedural version for every control type (`validate_phase350.py`).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from pymrst.utils.adi import value

BARSA = 1e5


def _total_rate(q_s: Sequence[Any]) -> Any:
    t = q_s[0]
    for i in range(1, len(q_s)):
        t = t + q_s[i]
    return t


def setup_well_control_equation_single_well(
    well: Any,
    bhp: Any,
    q_s: Sequence[Any],
    nph: int,
    mix_s: Optional[np.ndarray] = None,
    status: bool = True,
) -> Any:
    """The control equation for one well (MRST `setupWellControlEquationsSingleWell`).

    ``q_s`` are the surface phase rates (AD), ``bhp`` the well bottom-hole
    pressure (AD). ``mix_s`` is the per-phase surface mixture (to detect a
    singular single-phase control); when omitted every phase is assumed present.
    Phases are ordered water, oil, gas.
    """
    ctype = well.type
    val = well.val
    qt_s = _total_rate(q_s)

    if mix_s is None:
        mix_s = np.ones(nph)
    mix_s = np.abs(np.asarray(mix_s, dtype=float)).ravel()

    iw, io, ig = 0, 1, 2
    set_to_zero_rate = (val == 0) and ctype != "bhp"

    if not status:                                   # shut well
        return (bhp - value(bhp)) / BARSA if hasattr(bhp, "val") \
            else np.zeros_like(np.atleast_1d(bhp))
    if set_to_zero_rate:
        return qt_s

    if ctype == "bhp":
        return bhp - val
    if ctype in ("rate", "vrat"):
        return qt_s - val
    if ctype == "orat":
        return qt_s if mix_s[io] == 0 else q_s[io] - val
    if ctype == "wrat":
        return qt_s if mix_s[iw] == 0 else q_s[iw] - val
    if ctype == "grat":
        assert nph > 2, "a gas rate control needs a gas phase"
        return qt_s if mix_s[ig] == 0 else q_s[ig] - val
    if ctype == "lrat":
        singular = (mix_s[iw] + mix_s[io]) == 0
        return qt_s if singular else q_s[iw] + q_s[io] - val
    raise ValueError(f"unsupported well control '{ctype}' for this port")


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
setupWellControlEquationsSingleWell = setup_well_control_equation_single_well

__all__ = ["setup_well_control_equation_single_well"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.models.facilities.facility_model import (
        FacilityModel,
    )
    from pymrst.params import add_well, make_rock
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA as BAR, CENTI, DARCY, MILLI, POISE

    def _well(ctype, val, sign):
        g = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        init_simple_adi_fluid(mu=np.array([1.0, 5.0]) * CENTI * POISE,
                              rho=[1000.0, 800.0], n=[2, 2],
                              c=[1e-6 / BAR, 1e-4 / BAR], p_ref=200 * BAR)
        w = add_well(None, g, rock, np.array([0]), well_type=ctype, val=val,
                     compi=np.array([1.0, 0.0]), sign=sign)
        return w[0]

    def test_matches_facility_control_equation():
        """Reproduces FacilityModel._control_equation for every rate/bhp type."""
        nph = 2
        (qw, qo, bhp) = init_variables_adi(np.array([2e-3]), np.array([-1e-3]),
                                           np.array([210.0 * BARSA]))
        q_s = [qw, qo]
        for ctype, val, sign in [("bhp", 200 * BARSA, 1),
                                 ("rate", 3e-3, 1),
                                 ("wrat", 2e-3, 1),
                                 ("orat", 1e-3, -1),
                                 ("lrat", 1e-3, 1)]:
            w = _well(ctype, val, sign)
            new = setup_well_control_equation_single_well(w, bhp, q_s, nph)
            old = FacilityModel._control_equation(w, q_s, bhp, nph)
            assert np.allclose(value(new), value(old)), ctype
            if hasattr(new, "jac"):
                assert np.allclose((new - old).jac[0].toarray(), 0.0), ctype
        print("ok test_matches_facility_control_equation")

    def test_shut_and_singular_fallbacks():
        """A shut well and an absent-phase rate control fall back correctly."""
        nph = 2
        (qw, qo, bhp) = init_variables_adi(np.array([0.0]), np.array([0.0]),
                                           np.array([200.0 * BARSA]))
        q_s = [qw, qo]
        w = _well("wrat", 2e-3, 1)
        # water phase absent -> singular -> total-rate fallback
        eq = setup_well_control_equation_single_well(
            w, bhp, q_s, nph, mix_s=np.array([0.0, 1.0]))
        assert np.allclose(value(eq), value(qw + qo))
        # shut well -> trivial condition
        eq_shut = setup_well_control_equation_single_well(
            w, bhp, q_s, nph, status=False)
        assert np.allclose(value(eq_shut), 0.0)
        print("ok test_shut_and_singular_fallbacks")

    test_matches_facility_control_equation()
    test_shut_and_singular_fallbacks()
    print("\nAll tests passed!")
