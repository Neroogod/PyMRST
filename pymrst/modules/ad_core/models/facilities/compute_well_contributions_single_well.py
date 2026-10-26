"""One well's contributions to the reservoir equations (MRST
`computeWellContributionsSingleWell`).

The single-well counterpart of `compute_well_contributions_new`: same physics,
one well at a time, which is how `SimpleWell` and the facility model assemble.

Ported as its own module because the contract differs from what PyMRST already
had. `SimpleWell.computeWellEquations` returns
`(equations, control, [], [], q_mass, q_vol, wellSol)` and records `status` /
`cstatus` *on the wellSol*. MRST returns
`(eqs, cq_mass, mix_s, status, cstatus, cq_vol)` - and the difference that
matters is **`mix_s`**, the wellbore mixture composition, which nothing in this
port returned.

`mix_s` is what an outflowing perforation actually delivers when a well
crossflows: not the declared `compi`, but the mixture of the declared injection
stream and everything the inflowing perforations contributed. Without it a
caller has no way to ask what a crossflowing well is putting back into the
formation, and the answer is not recoverable from the equations afterwards -
they carry the summed source, not its composition.

The mixture itself comes from `cross_flow_mixture`; this is the per-well
assembly around it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.models.wells.compute_well_contributions_new import (
    compute_well_contributions_new,
)
from pymrst.utils.adi import value


def compute_well_contributions_single_well(
    well: Any, well_sol: Any, reservoir_model: Any, q_s: Sequence[Any],
    p_bh: Any, packed: Any,
) -> Tuple[list, list, np.ndarray, bool, np.ndarray, list]:
    """`(eqs, cq_mass, mix_s, status, cstatus, cq_vol)` for a single well.

    `packed` carries the perforation properties - pressure, mobilities,
    densities and dissolution ratios at the well's cells - as
    `pack_perforation_properties` builds them.
    """
    W = getattr(well, "W", well)
    n_perf = int(np.atleast_1d(np.asarray(W.WI)).size)
    n_phase = len(q_s)

    pressure, mobility, density, dissolved = _unpack(packed, n_perf, n_phase)
    rho_s = np.asarray(reservoir_model.get_surface_densities(),
                       dtype=float).ravel()
    b = [density[i]/rho_s[i] for i in range(n_phase)]

    wellmodel = SimpleNamespace(
        W=[W], referencePressure=pressure, bfactors=b,
        components=list(dissolved), mobilities=mobility,
        perf2well=np.zeros(n_perf, dtype=int),
        Rw=sp.csr_matrix(np.ones((n_perf, 1))),
        allowWellSignChange=bool(getattr(well, "allow_sign_change", False)),
        allowCrossflow=bool(getattr(well, "allow_crossflow", True)))

    eqs, cq_s, mix_s, status, cstatus, cq_r = compute_well_contributions_new(
        wellmodel, reservoir_model, [well_sol], p_bh, list(q_s))

    # MRST returns the connection fluxes in *mass*; the shared stack works in
    # surface volume, because its other caller wants that. Converting here
    # rather than inside keeps one implementation with two honest outputs.
    cq_mass = [cq_s[i]*rho_s[i] for i in range(n_phase)]
    return eqs, cq_mass, mix_s, bool(np.all(status)), cstatus, cq_r


computeWellContributionsSingleWell = compute_well_contributions_single_well


def _unpack(packed: Any, n_perf: int, n_phase: int):
    """Perforation properties, however `packed` spells them.

    A dict, a namespace, or the tuple `pack_perforation_properties` returns -
    all three occur at the call sites, and demanding one would make this
    usable from only some of them.
    """
    if isinstance(packed, (tuple, list)) and len(packed) >= 4:
        return packed[0], list(packed[1]), list(packed[2]), list(packed[3])

    pressure = _field(packed, "pressure", _field(packed, "p"))
    mobility = _field(packed, "mob", _field(packed, "mobility")) or []
    density = _field(packed, "rho", _field(packed, "density")) or []
    dissolved = _field(packed, "dissolved", []) or []

    if pressure is None:
        raise ValueError(
            "packed perforation properties carry no pressure; "
            "compute_well_contributions_single_well needs p, mob and rho")
    return pressure, list(mobility), list(density), list(dissolved)


def _field(obj: Any, name: str, default: Any = None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = ["compute_well_contributions_single_well",
           "computeWellContributionsSingleWell"]


if __name__ == "__main__":

    def _case(n_perf=2, sign=-1.0, crossflow=True):
        W = SimpleNamespace(WI=np.full(n_perf, 1e-11),
                            compi=np.array([1.0, 0.0]), status=True,
                            cstatus=np.ones(n_perf, dtype=bool), sign=sign)
        well = SimpleNamespace(W=W, allow_sign_change=False,
                               allow_crossflow=crossflow)
        sol = SimpleNamespace(cdp=np.zeros(n_perf),
                              cstatus=np.ones(n_perf, dtype=bool), status=True)
        packed = {"pressure": np.full(n_perf, 2.0e7),
                  "mob": [np.full(n_perf, 1e-3), np.full(n_perf, 1e-4)],
                  "rho": [np.full(n_perf, 1000.0), np.full(n_perf, 800.0)],
                  "dissolved": []}
        model = SimpleNamespace(
            disgas=False, vapoil=False,
            get_phase_names=lambda: ["W", "O"],
            get_surface_densities=lambda: np.array([1000.0, 800.0]))
        return well, sol, model, packed

    def test_it_returns_mrsts_six_values():
        """The contract that was missing: `computeWellEquations` returns a
        different 7-tuple and never yields `mix_s`."""
        well, sol, model, packed = _case()
        out = compute_well_contributions_single_well(
            well, sol, model, [np.array([-1e-3]), np.array([-1e-4])],
            np.array([1.9e7]), packed)
        assert len(out) == 6
        eqs, cq_mass, mix_s, status, cstatus, cq_vol = out
        assert len(eqs) == 2 and len(cq_mass) == 2 and len(cq_vol) == 2
        assert status is True and cstatus.all()
        print("ok test_it_returns_mrsts_six_values")

    def test_the_mixture_is_a_composition():
        """`mix_s` is what an outflowing perforation delivers under crossflow;
        it must be a composition."""
        well, sol, model, packed = _case()
        _, _, mix_s, *_ = compute_well_contributions_single_well(
            well, sol, model, [np.array([-1e-3]), np.array([-1e-4])],
            np.array([1.9e7]), packed)
        assert mix_s.shape == (1, 2)
        assert np.isclose(mix_s.sum(), 1.0), mix_s
        print(f"ok test_the_mixture_is_a_composition ({mix_s.ravel()})")

    def test_the_fluxes_are_mass_not_surface_volume():
        """MRST returns `cq_mass`; the shared stack works in surface volume,
        so the density weighting has to happen somewhere and it happens here."""
        well, sol, model, packed = _case()
        _, cq_mass, _, _, _, cq_vol = compute_well_contributions_single_well(
            well, sol, model, [np.array([-1e-3]), np.array([-1e-4])],
            np.array([1.9e7]), packed)
        ratio = np.asarray(cq_mass[0]).sum()/np.asarray(cq_vol[0]).sum()
        assert abs(ratio) > 1.0, "mass and volume fluxes are indistinguishable"
        print(f"ok test_the_fluxes_are_mass_not_surface_volume ({ratio:.1f})")

    def test_a_tuple_of_packed_properties_is_accepted():
        well, sol, model, packed = _case()
        as_tuple = (packed["pressure"], packed["mob"], packed["rho"], [])
        out = compute_well_contributions_single_well(
            well, sol, model, [np.array([-1e-3]), np.array([-1e-4])],
            np.array([1.9e7]), as_tuple)
        assert len(out) == 6
        print("ok test_a_tuple_of_packed_properties_is_accepted")

    test_it_returns_mrsts_six_values()
    test_the_mixture_is_a_composition()
    test_the_fluxes_are_mass_not_surface_volume()
    test_a_tuple_of_packed_properties_is_accepted()
    print("\nAll tests passed!")
