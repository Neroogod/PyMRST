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

import warnings

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

    def set_threshold_pressures_from_state(self, model: Any, state: Any
                                           ) -> "PhasePotentialDifferenceThresholded":
        """Derive thresholds that hold the *current* state in equilibrium.

        Used when a deck declares equilibration regions but no THPRES values:
        the thresholds are whatever it takes for the initial state not to
        flow. The construction is to evaluate the unthresholded potential
        difference and take, per face, the largest magnitude that must be
        overcome before flow can start in each direction.

        Faces whose upstream or downstream saturation is zero are excluded
        from their direction. A phase that is not present cannot flow, so its
        potential says nothing about what would hold it back, and including it
        gives a threshold set by a phase that is not there.
        """
        # `get_prop` returns one array per phase as a plain Python list, not a
        # stacked array - `value()` has to be mapped over it, not called on
        # the list itself (that hits `np.asarray` on a list of ADI objects,
        # which errors trying to broadcast them). `evaluate_on_domain` above
        # indexes the same property the same way (`pot[i]`), which is the
        # working pattern this follows.
        raw_potential = model.get_prop(state, "PhasePotentialDifference")
        phases = (raw_potential if isinstance(raw_potential, (list, tuple))
                  else [raw_potential])
        potential = np.column_stack([value(p) for p in phases]).astype(float)
        # `state["s"]` is not a field this port populates outside the Newton
        # update loop - only `sW`/`sG` are stored, and the closed-form phase
        # (`1 - sW - sG`) is derived, not stored. `Saturations` is the state
        # function that already does that derivation correctly for 1/2/3
        # phases (including the disgas/vapoil three-phase case with neither
        # saturation summing to 1 on its own); reusing it here is why this is
        # right instead of re-deriving the same closure a second, divergeable
        # way.
        from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
            Saturations,
        )
        s_list = Saturations(model).evaluate_on_domain(model, state)
        saturation = np.column_stack([value(s) for s in s_list]).astype(float)

        n_faces = potential.shape[0]
        # MRST's `operators.faceUpstr` (its own `faceUpstr.m`, closed over the
        # operators' `N`/`sz`). This port's `Operators.upw` is the same
        # two-argument upstream selection - `x[c1]` where `flag` else `x[c2]`
        # - under a different name; `faceUpstr` itself was never bound on
        # `Operators` at all, so this call raised `AttributeError` on any
        # model that actually reached it.
        upstream = model.operators.upw(np.ones(n_faces, dtype=bool),
                                       saturation)
        downstream = model.operators.upw(np.zeros(n_faces, dtype=bool),
                                         saturation)

        positive = np.where(np.asarray(downstream) == 0, np.nan, potential)
        negative = np.where(np.asarray(upstream) == 0, np.nan, potential)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pos_flow = np.nanmin(negative, axis=1)
            neg_flow = np.nanmax(positive, axis=1)

        # MRST's `setThresholdPressuresFromState`, translated literally rather
        # than re-derived: `pt(:,1) = posflow; pt(:,2) = negflow`, no
        # negation. A first version of this swapped the two columns *and*
        # negated one of them - it matched the docstring's prose ("largest
        # magnitude that must be overcome") but not MRST's actual formula,
        # and reapplying the derived threshold to the state it came from left
        # a residual of exactly the pressure drop (-4e6 Pa on the test case)
        # instead of the ~0 the docstring promises. That could only be found
        # by reading MRST's source, because this method had never been
        # called - four separate defects (an undefined attribute, `value()`
        # on a list, a state key this port never populates, and this sign/
        # column swap) had accumulated with nothing to catch any of them.
        #
        # MRST also only fills in entries that are still unset
        # (`isnan(pt(:,i))`), preserving any threshold already supplied (e.g.
        # from `from_deck`/THPRES) rather than clobbering it. This port's
        # `__init__` uses 0.0 rather than NaN as "unset" - matched here rather
        # than switching the whole class to a NaN sentinel, which would touch
        # `evaluate_on_domain`'s arithmetic on `self.pressure_threshold`.
        existing = self.pressure_threshold
        threshold = existing.copy()
        replace_pos = (existing[:, 0] == 0.0) & np.isfinite(pos_flow)
        replace_neg = (existing[:, 1] == 0.0) & np.isfinite(neg_flow)
        threshold[replace_pos, 0] = pos_flow[replace_pos]
        threshold[replace_neg, 1] = neg_flow[replace_neg]
        self.pressure_threshold = threshold
        return self

    setThresholdPressuresFromState = set_threshold_pressures_from_state

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
        GenericBlackOilModelAD,
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
        model = GenericBlackOilModelAD(g, rock, fluid, w, gravity=gravity)
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

    def test_thresholds_from_state_hold_the_current_state_in_equilibrium():
        """`set_threshold_pressures_from_state` - not exercised by any test
        above. It called `model.operators.faceUpstr`, an attribute `Operators`
        never had; on a real model this raised `AttributeError`
        unconditionally, and every test above passed anyway because none of
        them reach this method. Fixed to call `operators.upw`, this port's
        actual name for the same upstream-selection helper.

        The physics check, not just "doesn't crash": per the docstring, the
        derived thresholds must be exactly large enough to hold the state used
        to derive them in equilibrium - so re-evaluating the potential
        difference on that SAME state through the resulting thresholds must
        come back at (or under) zero everywhere.
        """
        model, st = _build_ow(gravity=9.80665)
        pot = PhasePotentialDifferenceThresholded(model)
        pot.set_threshold_pressures_from_state(model, st)
        model.FlowDiscretization.set_state_function(
            "PhasePotentialDifference", pot)
        st.clear_cache()
        out = model.get_props(st, "PhasePotentialDifference")
        for ph in out:
            vals = value(ph)
            assert np.all(np.abs(vals) < 1e-6), (
                "a threshold derived FROM this state must hold this state "
                f"still: {vals}")
        print("ok test_thresholds_from_state_hold_the_current_state_in_equilibrium")

    test_zero_threshold_matches_parent()
    test_threshold_blocks_subthreshold_flow()
    test_above_threshold_reduces_by_delta()
    test_directional_threshold_uses_correct_column()
    test_thresholds_from_state_hold_the_current_state_in_equilibrium()
    print("\nAll tests passed!")
