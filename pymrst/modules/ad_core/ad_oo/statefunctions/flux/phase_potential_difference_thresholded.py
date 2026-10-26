"""MRST ``ad-core/statefunctions/flux/PhasePotentialDifferenceThresholded.m``.

`PhasePotentialDifference` with the deck's threshold pressures (THPRES):
between two equilibration regions (EQLNUM) a phase only flows once its
potential difference exceeds the threshold, and then by the excess. Each
face keeps two thresholds, one per flow direction, from the THPRES records
(mirrored, unless EQLOPTS asks for IRREVERS); a potential difference whose
sign the threshold would flip becomes zero.

Defaulted (NaN) thresholds are set from a state by
`set_threshold_pressures_from_state`: the largest potential difference that
does not flow, per direction.
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.phase_potential_difference import (
    PhasePotentialDifference,
)
from pymrst.utils.adi import value


class PhasePotentialDifferenceThresholded(PhasePotentialDifference):
    """Phase potential differences reduced by the threshold pressures."""

    def __init__(self, model: Any):
        super().__init__(model)
        deck = model.inputdata
        imap = np.asarray(model.G.cells.indexMap, dtype=np.int64).ravel()
        equil = np.asarray(_field(deck, "REGIONS")["EQLNUM"]).ravel()[imap]
        thpres = np.atleast_2d(np.asarray(_field(deck, "SOLUTION")["THPRES"], dtype=float))
        eqlopts = _field(deck, "RUNSPEC").get("EQLOPTS") or []
        irreversible = any(str(o) == "IRREVERS" for o in eqlopts)
        if not irreversible:
            thpres = np.vstack([thpres, thpres[:, [1, 0, 2]]])
        N = np.asarray(model.operators.N, dtype=np.int64)
        threshold = np.zeros((N.shape[0], 2))
        left_equil = equil[N[:, 0]]
        right_equil = equil[N[:, 1]]
        for row in thpres:
            left = left_equil == row[0]
            right = right_equil == row[1]
            threshold[left & right, 0] = -row[2]
            left = left_equil == row[1]
            right = right_equil == row[0]
            threshold[left & right, 1] = row[2]
        if np.any(~np.isfinite(threshold)):
            warnings.warn("Non-defaulted threshold pressures detected. Please call "
                          "setThresholdPressuresFromState")
        self.pressureThreshold = threshold

    def set_threshold_pressures_from_state(self, model: Any, state: Any):
        """MRST `setThresholdPressuresFromState`: defaulted thresholds from `state`."""
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        pt = self.pressureThreshold.copy()
        model = copy(model)
        model.FlowDiscretization = copy(model.FlowDiscretization)
        model.FlowDiscretization.set_state_function("PhasePotentialDifference",
                                                    PhasePotentialDifference(model))
        state = model.validate_state(copy_state(state))
        state, _, _ = model.get_state_ad(state, True)
        pot = np.column_stack([np.asarray(value(p), dtype=float)
                               for p in model.get_prop(state, "PhasePotentialDifference")])
        nf = pot.shape[0]
        s = np.asarray(value(state["s"]), dtype=float)
        T = np.ones(nf, dtype=bool)
        s_up = model.operators.faceUpstr(T, s)
        s_down = model.operators.faceUpstr(~T, s)
        pot_pos, pot_neg = pot.copy(), pot.copy()
        pot_pos[s_down == 0] = np.nan
        pot_neg[s_up == 0] = np.nan
        posflow = np.nanmin(pot_neg, axis=1)
        negflow = np.nanmax(pot_pos, axis=1)
        assert np.all(np.isfinite(posflow))
        assert np.all(np.isfinite(negflow))
        replace_pos = np.isnan(pt[:, 0]) & np.isfinite(posflow)
        replace_neg = np.isnan(pt[:, 1]) & np.isfinite(negflow)
        pt[replace_pos, 0] = posflow[replace_pos]
        pt[replace_neg, 1] = negflow[replace_neg]
        pt[np.isnan(pt)] = 0
        self.pressureThreshold = pt
        return self

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        pot = super().evaluate_on_domain(model, state)
        thres = self.pressureThreshold
        for i in range(len(pot)):
            dpi = np.asarray(value(pot[i]), dtype=float)
            pos_flow = dpi < 0
            delta = thres[:, 1].copy()
            delta[pos_flow] = thres[pos_flow, 0]
            dp = pot[i] - delta
            ok = np.sign(dpi) == np.sign(np.asarray(value(dp)))
            pot[i] = dp*ok
        return pot

    setThresholdPressuresFromState = set_threshold_pressures_from_state
    evaluateOnDomain = evaluate_on_domain


def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


__all__ = ["PhasePotentialDifferenceThresholded"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def _thresholded(irrevers: bool = False):
        model = toy_model("WO", nx=4)
        model.inputdata = {"REGIONS": {"EQLNUM": np.array([1, 1, 2, 2])},
                           "SOLUTION": {"THPRES": np.array([[1, 2, 0.5]])},
                           "RUNSPEC": {"EQLOPTS": ["IRREVERS"] if irrevers else []}}
        return model, PhasePotentialDifferenceThresholded(model)

    def test_only_faces_between_regions_get_thresholds():
        _, ppd = _thresholded()
        np.testing.assert_allclose(ppd.pressureThreshold, [[0, 0], [-0.5, 0.5], [0, 0]])
        _, ppd = _thresholded(irrevers=True)
        np.testing.assert_allclose(ppd.pressureThreshold, [[0, 0], [-0.5, 0.0], [0, 0]])
        print("ok test_only_faces_between_regions_get_thresholds")

    def test_flow_needs_to_exceed_the_threshold():
        model, ppd = _thresholded()
        model.FlowDiscretization.set_state_function("PhasePotentialDifference", ppd)
        state = toy_state(model, pressure=np.array([3.0, 2.0, 1.8, 0.0]))
        pot = model.get_prop(state, "PhasePotentialDifference")
        # Face 2 drops 0.2 < 0.5 across the regions: no flow; faces 1, 3 untouched.
        np.testing.assert_allclose(pot[0], [-1.0, 0.0, -1.8])
        state = toy_state(model, pressure=np.array([3.0, 2.0, 1.0, 0.0]))
        pot = model.get_prop(state, "PhasePotentialDifference")
        np.testing.assert_allclose(pot[0], [-1.0, -0.5, -1.0])
        print("ok test_flow_needs_to_exceed_the_threshold")

    test_only_faces_between_regions_get_thresholds()
    test_flow_needs_to_exceed_the_threshold()
    print("\nAll tests passed!")
