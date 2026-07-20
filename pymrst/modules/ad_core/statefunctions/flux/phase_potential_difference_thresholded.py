"""PhasePotentialDifferenceThresholded (MRST ad-core/statefunctions/flux).

Ports MRST's `PhasePotentialDifferenceThresholded.m`: the phase potential
difference with ECLIPSE **threshold pressures** (`THPRES`) applied across
equilibration-region boundaries. A threshold pressure is the minimum potential
drop needed before a phase will flow across a sealing-but-not-quite fault: flow
starts only once the potential exceeds the threshold, and the threshold can be
different in the two directions (or irreversible).

The physics per face and phase, given the base potential difference ``dp``:

- pick the directional threshold ``delta`` (one value for ``dp >= 0``, another
  for ``dp < 0``);
- subtract it: ``dp' = dp - delta``;
- if subtracting the threshold would *reverse* the flow (the sign of ``dp'``
  differs from the sign of ``dp``), the drop did not clear the threshold, so
  the flow is blocked: ``dp' = 0``.

With a zero threshold this is exactly `PhasePotentialDifference`, so a model
that declares it but has no `THPRES` is undisturbed.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    PhasePotentialDifference,
)
from pymrst.utils.adi import value


class PhasePotentialDifferenceThresholded(PhasePotentialDifference):
    """Phase potential difference with directional threshold pressures.

    ``pressure_threshold`` is an ``(n_interior_faces, 2)`` array: column 0 is
    the threshold applied where the base potential is negative, column 1 where
    it is non-negative (MRST's two-column convention). ``None`` means no
    threshold - identical behaviour to the parent.
    """

    def __init__(self, model: Any = None,
                 pressure_threshold: Optional[np.ndarray] = None):
        super().__init__(model)
        if pressure_threshold is None and model is not None:
            n_if = model.operators.T.size
            pressure_threshold = np.zeros((n_if, 2))
        self.pressure_threshold = (None if pressure_threshold is None
                                   else np.asarray(pressure_threshold, float))

    @classmethod
    def from_deck(cls, model: Any, deck: Any
                  ) -> "PhasePotentialDifferenceThresholded":
        """Build the threshold array from a parsed ECLIPSE deck (``THPRES``).

        Mirrors MRST's constructor: match each face's two cells to their
        equilibration regions (``REGIONS.EQLNUM``) and look the pair up in
        ``SOLUTION.THPRES`` (region_a, region_b, pressure). Unless
        ``RUNSPEC.EQLOPTS`` contains ``IRREVERS`` the table is symmetric, so
        each entry is also applied in the reverse direction.
        """
        eqlnum = np.asarray(_deck_field(deck, "REGIONS", "EQLNUM"), int)
        index_map = getattr(model.g.cells, "index_map", None)
        if index_map is not None:
            eqlnum = eqlnum[np.asarray(index_map, int)]
        thpres = np.atleast_2d(np.asarray(_deck_field(deck, "SOLUTION",
                                                      "THPRES"), float))
        eqlopts = _deck_field(deck, "RUNSPEC", "EQLOPTS", default=[])
        irreversible = any(str(o).upper() == "IRREVERS" for o in np.atleast_1d(eqlopts))
        if not irreversible:
            rev = thpres[:, [1, 0, 2]]
            thpres = np.vstack([thpres, rev])

        n = model.operators.N
        nf = n.shape[0]
        threshold = np.zeros((nf, 2))
        left_r, right_r = eqlnum[n[:, 0]], eqlnum[n[:, 1]]
        for a, b, p in thpres:
            fwd = (left_r == a) & (right_r == b)
            threshold[fwd, 0] = -p
            rev = (left_r == b) & (right_r == a)
            threshold[rev, 1] = p
        return cls(model, threshold)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        pot = super().evaluate_on_domain(model, state)
        thres = self.pressure_threshold
        if thres is None:
            return pot
        out = []
        for i in range(len(pot)):
            dpi = value(pot[i])
            neg = dpi < 0
            delta = thres[:, 1].copy()
            delta[neg] = thres[neg, 0]
            dp = pot[i] - delta
            ok = (np.sign(dpi) == np.sign(value(dp))).astype(float)
            out.append(dp * ok)
        return out


def _deck_field(deck: Any, section: str, key: str, default: Any = None) -> Any:
    """Read ``deck[section][key]`` for either dict-like or attribute decks."""
    sec = deck.get(section) if isinstance(deck, dict) else getattr(deck, section, None)
    if sec is None:
        if default is not None:
            return default
        raise KeyError(f"deck has no {section} section")
    val = sec.get(key) if isinstance(sec, dict) else getattr(sec, key, None)
    if val is None:
        if default is not None:
            return default
        raise KeyError(f"deck {section} has no {key}")
    return val


__all__ = ["PhasePotentialDifferenceThresholded"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _build_ow(gravity=0.0):
        g = compute_geometry(cart_grid([6, 1, 1], [600.0, 100.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=1e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w, gravity=gravity)
        # a monotone pressure ramp so every interior face has a clear drop
        p = np.linspace(260, 160, g.cells.num) * BARSA
        st = State(pressure=p, sW=np.full(g.cells.num, 0.5),
                   qWs=np.array([1e-3]), qOs=np.array([-1e-3]),
                   bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st)

    def _register(model, thres):
        model.FlowDiscretization.set_state_function(
            "PhasePotentialDifference",
            PhasePotentialDifferenceThresholded(model, thres))

    def test_zero_threshold_matches_parent():
        """A zero threshold reproduces PhasePotentialDifference exactly."""
        model, st = _build_ow()
        base = [value(p).copy() for p in
                model.get_props(st, "PhasePotentialDifference")]
        st.clear_cache()
        _register(model, None)
        thr = model.get_props(st, "PhasePotentialDifference")
        for ph in range(model.n_phases):
            assert np.allclose(value(thr[ph]), base[ph])
        print("ok test_zero_threshold_matches_parent")

    def _signed_threshold(dpv, magnitude):
        """MRST convention: column 0 (dp<0) holds -mag, column 1 (dp>=0) +mag."""
        nf = dpv.size
        thres = np.zeros((nf, 2))
        thres[:, 0] = -magnitude       # applied where dp < 0
        thres[:, 1] = magnitude        # applied where dp >= 0
        return thres

    def test_threshold_blocks_subthreshold_flow():
        """A threshold larger than the drop blocks the phase entirely (0),
        in whichever direction the potential points."""
        model, st = _build_ow()
        base = model.get_props(st, "PhasePotentialDifference")
        dp0 = value(base[0])
        huge = np.max(np.abs(dp0)) * 10.0
        st.clear_cache()
        _register(model, _signed_threshold(dp0, huge))
        out = model.get_props(st, "PhasePotentialDifference")
        assert np.allclose(value(out[0]), 0.0), "sub-threshold flow not blocked"
        print("ok test_threshold_blocks_subthreshold_flow")

    def test_above_threshold_reduces_by_delta():
        """Above the threshold, the potential is pulled toward zero by exactly
        the threshold and keeps its sign; the Jacobian is unchanged."""
        model, st = _build_ow()
        base = model.get_props(st, "PhasePotentialDifference")
        dp0v = value(base[0])
        small = np.min(np.abs(dp0v)) * 0.25       # well below every drop
        thres = _signed_threshold(dp0v, small)
        st.clear_cache()
        _register(model, thres)
        out = model.get_props(st, "PhasePotentialDifference")
        # dp' = dp - signed_delta = dp pulled |small| toward zero, sign kept
        expect_val = dp0v - np.where(dp0v < 0, -small, small)
        assert np.allclose(value(out[0]), expect_val)
        # dp' differs from dp by a per-face constant, so the Jacobian is intact
        expect_ad = base[0] - np.where(dp0v < 0, -small, small)
        assert np.allclose((out[0] - expect_ad).jac[0].toarray(), 0.0)
        print("ok test_above_threshold_reduces_by_delta")

    def test_directional_threshold_uses_correct_column():
        """A face with a negative drop reads column 0, a positive one column 1."""
        model, st = _build_ow(gravity=9.80665)
        base = model.get_props(st, "PhasePotentialDifference")
        dpv = value(base[1])                       # oil potential, mixed signs possible
        nf = dpv.size
        thres = np.zeros((nf, 2))
        thres[:, 0] = -5 * BARSA                    # for dp < 0
        thres[:, 1] = 5 * BARSA                     # for dp >= 0
        st.clear_cache()
        _register(model, thres)
        out = value(model.get_props(st, "PhasePotentialDifference")[1])
        for f in range(nf):
            delta = thres[f, 0] if dpv[f] < 0 else thres[f, 1]
            expect = dpv[f] - delta
            if np.sign(dpv[f]) != np.sign(expect):
                expect = 0.0
            assert np.isclose(out[f], expect), (f, dpv[f], out[f], expect)
        print("ok test_directional_threshold_uses_correct_column")

    test_zero_threshold_matches_parent()
    test_threshold_blocks_subthreshold_flow()
    test_above_threshold_reduces_by_delta()
    test_directional_threshold_uses_correct_column()
    print("\nAll tests passed!")
