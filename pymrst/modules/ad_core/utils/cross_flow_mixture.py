"""Wellbore mixture composition under crossflow (MRST `crossFlowMixture`).

A well crossflows when some of its perforations take fluid *in* while others
put fluid *out*. What leaves an outflowing perforation is then not the
composition the well was told to inject, and not the composition of the cell it
enters - it is the composition of whatever is in the wellbore, which is the
declared injection stream plus everything the inflowing perforations
contributed.

That is the whole function:

    mixture = (inflow from perforations) + (net injection) * compi
    mixture = mixture / sum(mixture)

The alternative - giving every outflowing perforation the well's declared
`compi` - lets a producer that crossflows inject oil it never received. Mass
balances globally, because the same total leaves the well, but each phase is
wrong, and the error shows up as a saturation front moving where no fluid was
injected.

**Why `active` is guarded.** A well whose contributions cancel to exactly zero
has no mixture, and normalising would divide by zero. Its declared `compi` is
kept instead - the only defensible answer, since nothing is flowing to define
anything else.

**`conserve_mass`.** The default normalises by the mixture's own total, which
makes a composition summing to one. With `conserve_mass` the divisor is the
well's net injection instead, so the returned fractions reproduce the *rates*
rather than summing to one. MRST offers both because the two callers want
different things: an equation assembling volumes wants the second, a report
showing a composition wants the first.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def cross_flow_mixture(flux: np.ndarray, compi: np.ndarray, mapping: Any,
                       conserve_mass: bool = False,
                       is_zero: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-well injection composition, corrected for crossflow.

    `flux` is `(n_perforation, n_phase)` at surface conditions, negative into
    the wellbore. `compi` is `(n_well, n_phase)`, the declared composition.
    `mapping` carries `perf2well`. Returns a new `compi`; the input is not
    modified, because callers reuse it across Newton iterations.
    """
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    compi = np.atleast_2d(np.asarray(compi, dtype=float)).copy()

    # Inflow only: what each perforation delivers *into* the wellbore.
    flux_in = -np.minimum(flux, 0.0)
    if not np.any(flux_in):
        # Nothing crossflows, so the declared composition already describes
        # what leaves. Returning early is MRST's, and it matters: the
        # normalisation below would otherwise divide a zero mixture.
        return compi

    perf2well = np.asarray(_field(mapping, "perf2well"), dtype=int).ravel()
    n_well = compi.shape[0]

    net_flux = _sum_per_well(flux.sum(axis=1)[:, None], perf2well, n_well)
    net_injection = np.maximum(net_flux, 0.0)
    if is_zero is not None:
        # Wells the caller has already decided contribute nothing - a shut
        # well, or one whose rate unknown is being held fixed this iteration.
        net_injection[np.asarray(is_zero, dtype=bool)] = 0.0

    sum_in = _sum_per_well(flux_in, perf2well, n_well)
    top_in = net_injection*compi

    mixture = sum_in + top_in
    total = mixture.sum(axis=1, keepdims=True)

    active = total.ravel() > 0
    if np.any(active):
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = mixture/total
        compi[active, :] = normalised[active, :]

    if conserve_mass:
        # Divide by the well's net injection rather than the mixture total, so
        # the result reproduces rates instead of summing to one.
        injecting = top_in.sum(axis=1) > 0
        if np.any(injecting):
            divisor = np.maximum(net_flux[injecting], 0.0).sum(
                axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                compi[injecting, :] = (top_in[injecting, :]
                                       + sum_in[injecting, :])/divisor
    return compi


crossFlowMixture = cross_flow_mixture


def _sum_per_well(values: np.ndarray, perf2well: np.ndarray,
                  n_well: int) -> np.ndarray:
    """Accumulate a per-perforation array onto its wells (MRST `accumarray`)."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    out = np.zeros((n_well, values.shape[1]))
    for phase in range(values.shape[1]):
        np.add.at(out[:, phase], perf2well, values[:, phase])
    return out


def _field(obj: Any, name: str, default: Any = None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["cross_flow_mixture", "crossFlowMixture"]


if __name__ == "__main__":
    from types import SimpleNamespace

    MAP = SimpleNamespace(perf2well=np.array([0, 0]))

    def test_no_crossflow_leaves_the_declared_composition():
        """Every perforation injecting: nothing enters the wellbore, so the
        declared composition is what leaves."""
        flux = np.array([[1.0, 0.0], [2.0, 0.0]])
        compi = np.array([[1.0, 0.0]])
        out = cross_flow_mixture(flux, compi, MAP)
        assert np.allclose(out, [[1.0, 0.0]])
        print("ok test_no_crossflow_leaves_the_declared_composition")

    def test_inflow_changes_what_the_well_puts_back_out():
        """The point of the function. One perforation takes in pure oil while
        the well injects pure water; what leaves the other perforation is the
        mixture, not the declared water."""
        flux = np.array([[4.0, 0.0], [-0.0, -2.0]])   # out water / in oil
        compi = np.array([[1.0, 0.0]])
        out = cross_flow_mixture(flux, compi, MAP)
        assert out[0, 1] > 0.0, "oil taken in did not reach the mixture"
        assert np.isclose(out.sum(), 1.0), "composition must sum to one"
        print(f"ok test_inflow_changes_what_the_well_puts_back_out "
              f"(mixture {out[0, 0]:.3f}/{out[0, 1]:.3f})")

    def test_a_well_with_no_net_flow_keeps_its_composition():
        """Contributions cancelling exactly leaves nothing to define a
        mixture; normalising would divide by zero."""
        flux = np.array([[2.0, 0.0], [-2.0, 0.0]])
        compi = np.array([[0.3, 0.7]])
        out = cross_flow_mixture(flux, compi, MAP)
        assert np.all(np.isfinite(out))
        print("ok test_a_well_with_no_net_flow_keeps_its_composition")

    def test_the_input_is_not_modified():
        """Callers reuse `compi` across Newton iterations."""
        flux = np.array([[4.0, 0.0], [0.0, -2.0]])
        compi = np.array([[1.0, 0.0]])
        original = compi.copy()
        cross_flow_mixture(flux, compi, MAP)
        assert np.array_equal(compi, original)
        print("ok test_the_input_is_not_modified")

    test_no_crossflow_leaves_the_declared_composition()
    test_inflow_changes_what_the_well_puts_back_out()
    test_a_well_with_no_net_flow_keeps_its_composition()
    test_the_input_is_not_modified()
    print("\nAll tests passed!")
