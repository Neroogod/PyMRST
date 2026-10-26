"""MRST ``incomp/checkDrivingForcesIncomp.m``: whether an incompressible
solve has anything to do, warning when rates cannot balance (no bhp well,
no pressure boundary) or when there are no driving forces at all."""

import warnings
from typing import Any, Dict

import numpy as np


def check_driving_forces_incomp(g: Any, opt: Dict[str, Any]) -> bool:
    """MRST `do_solve = checkDrivingForcesIncomp(g, opt)`."""
    from pymrst.params.gravity.gravity import gravity
    bc, src, wells = opt.get("bc"), opt.get("src"), opt.get("wells") or []
    pressure_bc = bc is not None and any(str(t).lower() == "pressure"
                                         for t in np.atleast_1d(bc.type))
    well_bhp = len(wells) > 0 and any(str(w.type).lower() == "bhp" for w in wells)
    if not (pressure_bc or well_bhp):
        sum_rates = 0.0
        if wells:
            sum_rates += float(np.sum([np.sum(w.val) for w in wells]))
        if src is not None:
            sum_rates += float(np.sum(src.rate))
        if bc is not None:
            sum_rates += float(np.sum(bc.value))
        if abs(sum_rates) > 1e-12:
            warnings.warn("Well rates and flux bc must sum up to 0 \nwhen there are no bhp "
                          "constrained wells or pressure bc.Results may not be reliable.\n")
    g_vec = gravity()
    grav = np.linalg.norm(g_vec[:g.griddim]) > 0 or hasattr(g, "grav_pressure")
    do_solve = grav or not all([bc is None, src is None, len(wells) == 0, opt.get("bcp") is None])
    do_solve = do_solve or bool(opt.get("MatrixOutput", False))
    if "rhs" in opt:
        do_solve = do_solve or opt["rhs"] is not None
    if not do_solve:
        warnings.warn("No external driving forces present in model--state remains unchanged.\n")
    return bool(do_solve)


checkDrivingForcesIncomp = check_driving_forces_incomp
__all__ = ["check_driving_forces_incomp", "checkDrivingForcesIncomp"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.wells_and_bc import add_source

    G = compute_geometry(cart_grid([2, 2, 1]))

    def test_nothing_to_do_without_forces_or_gravity():
        with isolated_gravity("off"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not check_driving_forces_incomp(G, {})
        assert any("No external driving forces" in str(w.message) for w in caught)
        print("ok test_nothing_to_do_without_forces_or_gravity")

    def test_unbalanced_sources_warn_but_solve():
        src = add_source(None, np.array([0]), 1.0)
        with isolated_gravity("off"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert check_driving_forces_incomp(G, {"src": src})
        assert any("must sum up to 0" in str(w.message) for w in caught)
        print("ok test_unbalanced_sources_warn_but_solve")

    def test_gravity_alone_is_a_driving_force():
        with isolated_gravity("reset", "on"):
            assert check_driving_forces_incomp(G, {})
        print("ok test_gravity_alone_is_a_driving_force")

    test_nothing_to_do_without_forces_or_gravity()
    test_unbalanced_sources_warn_but_solve()
    test_gravity_alone_is_a_driving_force()
    print("\nAll tests passed!")
