"""Well control switching (MRST ad-core/models/wells).

Ports MRST's `updateSwitchedControls.m` / `updateSwitchedWellControls.m` (AC13):
a well runs on one control until it breaks a limit, then switches to the limit
it broke. A rate-controlled injector that would need more than its maximum
bottom-hole pressure switches to hold that pressure; a bhp-controlled producer
that exceeds its rate limit switches to hold the rate. Without this a schedule's
"inject at rate unless pressure exceeds X" cannot be expressed.

Rates are signed (production negative), so a producer breaks a rate limit by
going *below* it - which is why the producer comparisons read upside down.

PyMRST does this inside `FacilityModel.update_switched_controls` /
`_violated_limit` (MATLAB-validated, Phase 39). This module is the same logic as
a standalone function on MRST's layout, validated to switch the same wells to
the same modes (`validate_phase350.py`).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def violated_limit(well: Any, bhp: float, q: np.ndarray) -> Optional[str]:
    """The first limit ``well`` breaks other than the one it is on, or None."""
    lims = getattr(well, "lims", None)
    if not lims:
        return None
    qt = float(np.asarray(q, float).sum())

    if well.sign > 0:                                    # injector
        checks = [("bhp", bhp > lims.get("bhp", np.inf)),
                  ("rate", qt > lims.get("rate", np.inf))]
    else:                                                # producer
        q = np.asarray(q, float)
        q_w = q[0]
        q_o = q[1] if q.size > 1 else 0.0
        q_g = q[2] if q.size > 2 else 0.0
        checks = [
            ("bhp", bhp < lims.get("bhp", -np.inf)),
            ("orat", q_o < lims.get("orat", -np.inf)),
            ("lrat", q_w + q_o < lims.get("lrat", -np.inf)),
            ("grat", q_g < lims.get("grat", -np.inf)),
            ("wrat", q_w < lims.get("wrat", -np.inf)),
            ("vrat", qt > -lims.get("vrat", np.inf)),
        ]
    for mode, is_violated in checks:
        if mode == well.type:                # never switch to the current mode
            continue
        if is_violated and mode in lims:
            return mode
    return None


def update_switched_controls(wells: Sequence[Any], bhp: np.ndarray,
                             rates: np.ndarray, verbose: bool = False
                             ) -> Tuple[bool, List[Tuple[int, str, str]]]:
    """Switch each well that has violated a limit.

    ``bhp`` is per-well, ``rates`` is ``(n_well, n_phase)`` surface rates.
    Mutates each well's ``type``/``val`` to the broken limit. Returns
    ``(all_within_limits, switches)`` where ``switches`` lists
    ``(well_index, from_type, to_type)``.
    """
    bhp = np.asarray(bhp, float).ravel()
    rates = np.atleast_2d(np.asarray(rates, float))
    all_ok = True
    switches: List[Tuple[int, str, str]] = []
    for k, w in enumerate(wells):
        if not getattr(w, "status", True) or not getattr(w, "lims", None):
            continue
        mode = violated_limit(w, bhp[k], rates[k])
        if mode is None:
            continue
        all_ok = False
        switches.append((k, w.type, mode))
        if verbose:
            print(f"  well {getattr(w, 'name', k)}: {w.type} -> {mode}")
        w.type = mode
        w.val = float(w.lims[mode])
    return all_ok, switches


__all__ = ["update_switched_controls", "violated_limit"]


if __name__ == "__main__":
    class _W:
        def __init__(self, name, ctype, val, sign, lims):
            self.name, self.type, self.val = name, ctype, val
            self.sign, self.lims, self.status = sign, lims, True

    def test_injector_switches_to_bhp_on_pressure_limit():
        """A rate injector needing too much pressure switches to hold bhp."""
        w = _W("I", "rate", 3e-3, +1, {"bhp": 300e5, "rate": 3e-3})
        ok, sw = update_switched_controls([w], np.array([320e5]),
                                          np.array([[3e-3, 0.0]]))
        assert not ok and sw == [(0, "rate", "bhp")]
        assert w.type == "bhp" and w.val == 300e5
        print("ok test_injector_switches_to_bhp_on_pressure_limit")

    def test_producer_switches_on_rate_limit():
        """A bhp producer exceeding its oil-rate limit switches to hold orat."""
        w = _W("P", "bhp", 150e5, -1, {"orat": -1e-3})
        ok, sw = update_switched_controls([w], np.array([150e5]),
                                          np.array([[0.0, -2e-3]]))
        assert not ok and sw == [(0, "bhp", "orat")]
        assert w.type == "orat"
        print("ok test_producer_switches_on_rate_limit")

    def test_within_limits_does_not_switch():
        """A well inside every limit keeps its control."""
        w = _W("I", "rate", 3e-3, +1, {"bhp": 400e5})
        ok, sw = update_switched_controls([w], np.array([300e5]),
                                          np.array([[3e-3, 0.0]]))
        assert ok and sw == [] and w.type == "rate"
        print("ok test_within_limits_does_not_switch")

    def test_never_switches_to_current_mode():
        """A well on bhp does not 'switch' to bhp even if the bhp limit binds."""
        w = _W("I", "bhp", 300e5, +1, {"bhp": 300e5})
        ok, sw = update_switched_controls([w], np.array([320e5]),
                                          np.array([[3e-3, 0.0]]))
        assert ok and sw == []
        print("ok test_never_switches_to_current_mode")

    test_injector_switches_to_bhp_on_pressure_limit()
    test_producer_switches_on_rate_limit()
    test_within_limits_does_not_switch()
    test_never_switches_to_current_mode()
    print("\nAll tests passed!")
