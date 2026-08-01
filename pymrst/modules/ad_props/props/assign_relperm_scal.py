"""End-point-scaled relative permeability (MRST ``props/assignRelPermScal.m``).

The same dispatcher as `assign_relperm`, except that saturations are rescaled
onto the tables' own end points before lookup. A deck that varies connate
water or residual oil cell by cell states one table and scales it, rather than
tabulating a curve per cell.

Each phase is normalised onto its mobile range, for example for water

    Sw* = (Sw - Swcr) / (Swu - Sowcr - Swcr)

so Sw = Swcr maps to 0 and the table's own abscissa is recovered.

**MRST's three-phase branch does not run.** Three defects, all visible in
`assignRelPermScal.m`:

1. the dispatcher builds ``@(sw, sg, sgmax, varargin) relPermWOG(sw, sg, f,
   opt, varargin{:})`` while the callee is ``relPermWOG(sw, sg, sgmax, f,
   opt, ...)``, so ``f`` arrives where ``sgmax`` is expected;
2. the imbibition term references an undefined ``sg_max`` (the argument is
   spelled ``sgmax``), which is an immediate error;
3. ``krG(sg<sgmax) = krG_inem`` assigns a full-length vector into a subset,
   which only works when every cell happens to satisfy the mask.

This port implements the evident intent - argument order fixed, ``sgmax``
used, the imbibition value taken on the masked subset - so the three-phase
path works here. It therefore *differs from MRST*, because matching MRST
exactly would mean reproducing a crash. The two-phase branches are unaffected
and are straight ports.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pymrst.utils.adi import value


@dataclass
class ScalingPoints:
    """Per-cell saturation end points, MRST's ``opt`` for this function."""

    swcr: Any = 0.0        # critical water
    swu: Any = 1.0         # maximum water
    sowcr: Any = 0.0       # residual oil to water
    sgcr: Any = 0.0        # critical gas
    sgu: Any = 1.0         # maximum gas
    sogcr: Any = 0.0       # residual oil to gas
    sgl: Any = 0.0         # connate gas
    isgcr: Any = 0.0       # imbibition counterparts
    isgu: Any = 1.0
    isogcr: Any = 0.0
    isgl: Any = 0.0


def _one(curve, name: str):
    if callable(curve):
        return curve
    if isinstance(curve, (list, tuple)) and len(curve) == 1:
        return curve[0]
    raise ValueError(f"relPermScal needs a single {name} curve")


def assign_relperm_scal(f: Dict[str, Any],
                        opt: Optional[ScalingPoints] = None) -> Dict[str, Any]:
    """Attach a ``relPerm`` that scales saturations onto the tables."""
    opt = opt or ScalingPoints()
    has_og, has_ow = "krOG" in f, "krOW" in f

    if not has_og:                                   # water/oil
        kr_w = _one(f["krW"], "krW")
        kr_o = _one(f.get("krO") or f["krOW"], "krO")

        def rel_perm(sw, *args, **kwargs):
            return kr_w(sw, *args, **kwargs), kr_o(1 - sw, *args, **kwargs)

    elif not has_ow:                                 # oil/gas
        kr_g = _one(f["krG"], "krG")
        kr_o = _one(f.get("krO") or f["krOG"], "krO")

        def rel_perm(sg, *args, **kwargs):
            return kr_o(1 - sg, *args, **kwargs), kr_g(sg, *args, **kwargs)

    else:                                            # three-phase
        kr_w = _one(f["krW"], "krW")
        kr_g = _one(f["krG"], "krG")
        kr_ow = _one(f["krOW"], "krOW")
        kr_og = _one(f["krOG"], "krOG")

        def scaled_gas(sg):
            return (sg - opt.sgcr) / (opt.sgu - opt.sogcr - opt.sgcr)

        def rel_perm(sw, sg, sgmax, *args, **kwargs):
            weight = (sw - opt.swcr) / (sg + sw - opt.swcr)

            sw_m = (sw - opt.swcr) / (opt.swu - opt.sowcr - opt.swcr)
            kr_water = kr_w(sw_m, *args, **kwargs)

            kr_gas = kr_g(scaled_gas(sg), *args, **kwargs)

            # Imbibition: below the historical maximum gas saturation the gas
            # curve is re-entered on the imbibition end points.
            span = opt.isgu - opt.isogcr - opt.isgcr
            sg_in_max = scaled_gas(sgmax) * span + opt.isgcr
            imbibition = np.maximum(
                kr_g((sg_in_max - (sgmax - sg) - opt.isgcr) / span,
                     *args, **kwargs), 0.0)
            below = value(sg) < value(sgmax)
            if np.any(below):
                kr_gas = np.where(below, imbibition, kr_gas)

            so = 1 - sw - sg
            som_w = ((so - opt.sowcr)
                     / (1 - opt.swcr - opt.sgl - opt.sowcr))
            som_g = ((so - opt.sogcr)
                     / (1 - opt.isgl - opt.swcr - opt.sgl - opt.sogcr))
            kr_oil = ((1 - weight) * kr_og(som_g, *args, **kwargs)
                      + weight * kr_ow(som_w, *args, **kwargs))
            return kr_water, kr_oil, kr_gas

    f["relPerm"] = rel_perm
    return f


assignRelPermScal = assign_relperm_scal


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
    from pymrst.modules.ad_props.props.assign_swof import assign_swof
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    SWOF = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.50, 0.20, 0.40, 0.0],
                     [1.00, 1.00, 0.00, 0.0]])
    SGOF = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.40, 0.30, 0.35, 0.0],
                     [1.00, 1.00, 0.00, 0.0]])

    def _fluid():
        reg = Regions()
        f = assign_swof({}, [SWOF], reg)
        f = assign_sgof(f, [SGOF], reg)
        for name in ("krW", "krOW", "krG", "krOG"):
            f[name] = f[name][0]
        return f

    def test_water_oil_returns_two_curves() -> None:
        f = assign_swof({}, [SWOF], Regions())
        f["krW"], f["krOW"] = f["krW"][0], f["krOW"][0]
        f = assign_relperm_scal(f)
        kr_w, kr_o = f["relPerm"](np.array([0.50]))
        assert np.isclose(kr_w, 0.20) and np.isclose(kr_o, 0.40)
        print("ok test_water_oil_returns_two_curves")

    def test_default_points_leave_saturations_unscaled() -> None:
        """With the identity end points, scaling must be a no-op."""
        f = assign_relperm_scal(_fluid())
        sw, sg = np.array([0.50]), np.array([0.0])
        kr_w, _, _ = f["relPerm"](sw, sg, sg)
        assert np.isclose(kr_w, 0.20), kr_w
        print("ok test_default_points_leave_saturations_unscaled")

    def test_critical_water_shifts_the_water_curve() -> None:
        opt = ScalingPoints(swcr=0.2, swu=1.0, sowcr=0.0)
        f = assign_relperm_scal(_fluid(), opt)
        sw, sg = np.array([0.6]), np.array([0.0])
        kr_w, _, _ = f["relPerm"](sw, sg, sg)
        # Sw* = (0.6 - 0.2) / (1.0 - 0.0 - 0.2) = 0.5 -> krW(0.5) = 0.20
        assert np.isclose(kr_w, 0.20), kr_w
        print("ok test_critical_water_shifts_the_water_curve")

    def test_scaling_actually_changes_the_answer() -> None:
        base = assign_relperm_scal(_fluid())
        scaled = assign_relperm_scal(_fluid(), ScalingPoints(swcr=0.2))
        sw, sg = np.array([0.6]), np.array([0.0])
        assert not np.isclose(base["relPerm"](sw, sg, sg)[0],
                              scaled["relPerm"](sw, sg, sg)[0])
        print("ok test_scaling_actually_changes_the_answer")

    def test_three_phase_runs_and_returns_three_curves() -> None:
        """MRST's own three-phase branch errors here; this port's does not."""
        f = assign_relperm_scal(_fluid())
        out = f["relPerm"](np.array([0.3]), np.array([0.2]), np.array([0.2]))
        assert len(out) == 3 and all(np.all(np.isfinite(v)) for v in out)
        print("ok test_three_phase_runs_and_returns_three_curves")

    def test_matching_end_points_mean_no_hysteresis() -> None:
        """The imbibition curve is the drainage one when the points agree.

        With identical drainage and imbibition end points the scaling maps a
        cell back onto the same abscissa, so a cell below its historical
        maximum reads the same value. Anything else would be hysteresis
        appearing from nowhere.
        """
        f = assign_relperm_scal(_fluid())
        sw, sg = np.array([0.3, 0.3]), np.array([0.2, 0.2])
        _, _, kr_g = f["relPerm"](sw, sg, np.array([0.2, 0.5]))
        assert np.isclose(kr_g[0], kr_g[1]), kr_g
        print("ok test_matching_end_points_mean_no_hysteresis")

    def test_differing_imbibition_points_produce_hysteresis() -> None:
        opt = ScalingPoints(sgcr=0.0, sgu=1.0, isgcr=0.05, isgu=0.9)
        f = assign_relperm_scal(_fluid(), opt)
        sw, sg = np.array([0.3, 0.3]), np.array([0.2, 0.2])
        _, _, kr_g = f["relPerm"](sw, sg, np.array([0.2, 0.5]))
        assert not np.isclose(kr_g[0], kr_g[1]), kr_g
        print("ok test_differing_imbibition_points_produce_hysteresis")

    def test_the_imbibition_branch_is_only_taken_where_it_applies() -> None:
        opt = ScalingPoints(isgcr=0.05, isgu=0.9)
        f = assign_relperm_scal(_fluid(), opt)
        sw, sg = np.array([0.3, 0.3]), np.array([0.2, 0.2])
        equal = f["relPerm"](sw, sg, sg)[2]
        drained = f["relPerm"](sw, sg, np.array([0.2, 0.6]))[2]
        assert np.isclose(equal[0], drained[0]), "untouched cell must not move"
        assert not np.isclose(equal[1], drained[1]), "drained cell must move"
        print("ok test_the_imbibition_branch_is_only_taken_where_it_applies")

    def test_relative_permeability_never_goes_negative() -> None:
        f = assign_relperm_scal(_fluid())
        sw, sg = np.array([0.3]), np.array([0.05])
        out = f["relPerm"](sw, sg, np.array([0.9]))
        assert all(np.all(v >= 0.0) for v in out), out
        print("ok test_relative_permeability_never_goes_negative")

    test_water_oil_returns_two_curves()
    test_default_points_leave_saturations_unscaled()
    test_critical_water_shifts_the_water_curve()
    test_scaling_actually_changes_the_answer()
    test_three_phase_runs_and_returns_three_curves()
    test_matching_end_points_mean_no_hysteresis()
    test_differing_imbibition_points_produce_hysteresis()
    test_the_imbibition_branch_is_only_taken_where_it_applies()
    test_relative_permeability_never_goes_negative()
    print("\nAll tests passed!")
