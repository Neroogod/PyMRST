"""Phase-appearance flash after a Newton step (MRST ``computeFlashBlackOil``).

Written because F0.4's risk ranking put this module among the highest-scoring
files with no focused test: it decides phase appearance and writes the
hydrocarbon status every downstream property switches on, and a wrong answer
here is silent - the saturations still sum to one and the run continues.
"""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_blackoil.utils.compute_flash_black_oil import (
    compute_flash_black_oil,
)


class _Fluid:
    """Flat saturation curves, so a test's numbers come from the flash."""

    def rs_sat(self, pressure):
        return np.full(np.size(pressure), 50.0)

    def rv_sat(self, pressure):
        return np.full(np.size(pressure), 1.0)


def _model(*, disgas=True, vapoil=False):
    return SimpleNamespace(raw_fluid=_Fluid(), disgas=disgas, vapoil=vapoil)


def _state(sw, so, sg, *, rs=10.0, rv=0.0, status=0, pressure=2.0e7):
    count = len(sw)
    return {
        "sW": np.array(sw, dtype=float),
        "sO": np.array(so, dtype=float),
        "sG": np.array(sg, dtype=float),
        "rs": np.full(count, float(rs)),
        "rv": np.full(count, float(rv)),
        "pressure": np.full(count, float(pressure)),
        "status": np.full(count, int(status), dtype=np.int64),
    }


def _saturations(state):
    return np.column_stack([state["sW"], state["sO"], state["sG"]])


def test_negative_gas_is_clipped_and_the_others_take_its_share():
    """The clipping cascade, which is where MRST puts the phase bookkeeping.

    A negative saturation is not rounded to zero - the remaining phases are
    rescaled by ``1 - s_negative``, which is what keeps the triple summing to
    one without inventing volume.
    """
    state = compute_flash_black_oil(
        _state([0.2], [0.9], [-0.1]), _state([0.2], [0.8], [0.0]), _model(),
    )
    np.testing.assert_allclose(
        _saturations(state), [[0.2/1.1, 0.9/1.1, 0.0]], rtol=1e-12,
    )
    assert state["sG"][0] == 0.0


def test_each_phase_in_turn_can_go_negative():
    for index, sats in enumerate(([-0.1, 0.6, 0.5],
                                  [0.3, -0.1, 0.8],
                                  [0.3, 0.8, -0.1])):
        state = compute_flash_black_oil(
            _state([sats[0]], [sats[1]], [sats[2]]),
            _state([0.3], [0.4], [0.3]), _model(),
        )
        row = _saturations(state)[0]
        assert np.all(row >= 0.0), (index, row)
        np.testing.assert_allclose(row.sum(), 1.0, rtol=1e-12)


def test_rs_is_pinned_to_the_curve_where_gas_is_present_and_capped_elsewhere():
    """The asymmetry MRST is explicit about, and it is easy to get backwards.

    Where gas is present `rs` is *set* to the saturated value; where it is not,
    `rs` is only *capped* by it. Setting it in both cases would invent
    dissolved gas in an undersaturated cell.
    """
    # cell 0: gas present (sg > 0), rs below the curve -> pinned up to 50
    # cell 1: no gas, rs above the curve -> capped down to 50
    # cell 2: no gas, rs below the curve -> left alone
    state = compute_flash_black_oil(
        _state([0.1, 0.1, 0.1], [0.6, 0.9, 0.9], [0.3, 0.0, 0.0],
               rs=10.0, status=1),
        _state([0.1, 0.1, 0.1], [0.6, 0.9, 0.9], [0.3, 0.0, 0.0], status=1),
        _model(),
    )
    state["rs"][:] = state["rs"]
    result = compute_flash_black_oil(
        {**_state([0.1, 0.1, 0.1], [0.6, 0.9, 0.9], [0.3, 0.0, 0.0], status=0),
         "rs": np.array([10.0, 90.0, 10.0])},
        _state([0.1, 0.1, 0.1], [0.6, 0.9, 0.9], [0.3, 0.0, 0.0], status=0),
        _model(),
    )
    np.testing.assert_allclose(result["rs"], [50.0, 50.0, 10.0])


def test_rs_never_leaves_the_flash_negative():
    """Needs a cell where `rs` survives to the end to be clamped.

    The obvious version of this test - any cell with a negative `rs` - does
    not test the clamp at all: `rs == 0` is false but `sg > 0` makes gas
    present, and `rs` is then *set* to the saturated value before the clamp is
    reached. It has to be a cell with no free gas whose previous status keeps
    gas absent, so `rs` is only capped and stays negative until `max(rs, 0)`.
    """
    result = compute_flash_black_oil(
        {**_state([0.2], [0.8], [0.0], status=1), "rs": np.array([-5.0])},
        {**_state([0.2], [0.8], [0.0], status=1), "rs": np.array([-5.0])},
        _model(),
    )
    assert int(result["status"][0]) == 1, "this cell must have no free gas"
    assert result["rs"][0] == 0.0


def test_status_packs_oil_and_gas_presence_into_one_integer():
    """`status = oil_present + 2*gas_present`, which downstream code decodes.

    Pinned because the factor of two carries the gas bit: dropping it makes
    an oil-and-gas cell indistinguishable from an oil-only one.
    """
    result = compute_flash_black_oil(
        _state([0.1], [0.6], [0.3], status=0), _state([0.1], [0.6], [0.3]),
        _model(vapoil=False),
    )
    # vapoil off means oil is always present, and sg > 0 means gas is too.
    assert int(result["status"][0]) == 1 + 2*1


def test_without_dissolved_gas_the_previous_rs_is_the_bound_and_gas_is_present():
    """MRST's ``~disgas`` branch, which this port omitted entirely.

    Without dissolved gas there is no saturation curve to cross: `rsSat`
    becomes the previous `rs`, and `gasPresent` is true everywhere. The
    omission was reachable through `disgas=False, vapoil=True` - the caller in
    `update_state_black_oil_generic` skips the flash only when *both* are
    false - and there it reported status 1 where MRST reports 3, because
    `gasPresent` is what contributes the 2.
    """
    result = compute_flash_black_oil(
        _state([0.2], [0.7], [0.1], rs=10.0, status=1),
        _state([0.2], [0.7], [0.1], rs=10.0, status=1),
        _model(disgas=False, vapoil=True),
    )
    assert int(result["status"][0]) == 3
    # rs is bounded by the previous rs, not by any curve the fluid exposes.
    np.testing.assert_allclose(result["rs"], [10.0])


def test_without_dissolved_gas_a_high_rs_is_pulled_back_to_the_previous_value():
    result = compute_flash_black_oil(
        {**_state([0.2], [0.7], [0.1], status=1), "rs": np.array([900.0])},
        _state([0.2], [0.7], [0.1], rs=10.0, status=1),
        _model(disgas=False, vapoil=True),
    )
    np.testing.assert_allclose(result["rs"], [10.0])


def test_a_water_only_cell_reports_both_phases_present():
    """`watOnly` forces presence, so an empty cell does not look undersaturated."""
    result = compute_flash_black_oil(
        _state([1.0], [0.0], [0.0], status=1), _state([1.0], [0.0], [0.0]),
        _model(),
    )
    assert int(result["status"][0]) == 3


def test_saturations_are_renormalized_which_mrst_does_not_do():
    """A deliberate departure, pinned so it stays deliberate.

    MRST writes the saturations as the clipping cascade leaves them. On a
    well-formed state the two agree exactly, because each clip divides by
    ``1 - s_negative`` and so preserves a unit sum. This only bites on input
    that did not sum to one, where MRST propagates the error and PyMRST
    repairs it - which also means this function cannot be used to detect that
    upstream fault.
    """
    result = compute_flash_black_oil(
        _state([0.2], [0.5], [0.1]), _state([0.2], [0.5], [0.1]), _model(),
    )
    np.testing.assert_allclose(_saturations(result).sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(_saturations(result), [[0.25, 0.625, 0.125]])


def test_the_clipping_cascade_and_a_plain_renormalization_agree():
    """Why no test here can detect a broken cascade, and that is not a gap.

    Deleting the rescaling and letting the final renormalization do the work
    was tried as a mutation and no test caught it. The reason is arithmetic,
    not coverage: for input summing to one the two are the same operation.
    Clipping `sg` and dividing the rest by `1 - sg` gives exactly what
    zeroing `sg` and dividing by the new sum gives, and the same holds when
    two phases clip in sequence.

    So the cascade is redundant *given* the renormalization, and only the
    renormalization is load-bearing. Recorded here so the next person does not
    read the surviving mutant as missing coverage.
    """
    for sats in ([0.2, 0.9, -0.1], [-0.1, 1.2, -0.1], [0.3, -0.2, 0.9]):
        cascaded = compute_flash_black_oil(
            _state([sats[0]], [sats[1]], [sats[2]]),
            _state([0.3], [0.4], [0.3]), _model(),
        )
        clamped = np.maximum(np.array(sats, dtype=float), 0.0)
        np.testing.assert_allclose(
            _saturations(cascaded)[0], clamped/clamped.sum(), rtol=1e-12,
            err_msg=f"cascade and renormalization disagree for {sats}",
        )
