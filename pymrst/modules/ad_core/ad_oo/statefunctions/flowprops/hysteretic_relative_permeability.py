"""MRST ``ad-core/statefunctions/flowprops/hysteresis/HystereticRelativePermeability.m``.

`BaseRelativePermeability` with gas-phase hysteresis: the gas relative
permeability is evaluated against the historic maximum gas saturation
(`sgMax`, from the state's `sMax`). In the imbibition regions
(`rock.regions.imbibition`) that the fluid marks hysteretic (`krHyst`), a
cell whose gas saturation has fallen below its maximum is on a Killough
(1976) scanning curve between the drainage curve `krG{reg}` and the
imbibition curve `krG{reg + nregi}` (`scanRelPerm`); elsewhere it is on the
region's drainage curve. The end points come from `krPts.g` (or `sWcon`).

As in MRST, `krHyst` (1 when EHYSTR asks for relative-permeability
hysteresis) is compared with the imbibition region number, so only region 1
can be hysteretic, and the gas curve is evaluated without end-point scaling.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.base_relative_permeability import (
    BaseRelativePermeability,
)
from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis.evaluate_function_cell_subset_reg import (
    evaluate_function_cell_subset_reg,
)
from pymrst.modules.ad_core.statefunctions.state_function import STATE
from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad
from pymrst.utils.adi import ad_max, value

SCAN_TOLERANCE: float = 1e-3
KILLOUGH: int = 2


class HystereticRelativePermeability(BaseRelativePermeability):
    """Relative permeability with gas hysteresis on imbibition."""

    def __init__(self, model: Any, regions: Any = None, **options: Any):
        super().__init__(model, regions, **options)
        self.sg_min = None
        self.sg_max = None
        self.sg_tmax = None
        self.nreg_imbibition = 1
        self.minSat = 0.05
        nc = model.G.cells.num
        rock_regions = _rock_regions(model)
        if self.regions is None:
            self.regions = (np.asarray(rock_regions["saturation"]).ravel()
                            if "saturation" in rock_regions else np.ones(nc))
        self.depends_on(["s", "sMax"], STATE)
        regi = np.asarray(rock_regions.get("imbibition", np.full(nc, 2))).ravel()
        regi = np.unique(regi).astype(np.int64)
        nregi = regi.size
        reg = regi - nregi
        f = model.fluid
        if getattr(f, "sWcon", None) is not None:
            sgmax = 1 - np.atleast_1d(np.asarray(f.sWcon, dtype=float))
            if sgmax.size != nregi:
                sgmax = np.ones(nregi)*sgmax
            sgmin = np.zeros(nregi)
            sgtmax = sgmax
            warnings.warn("Min gas saturation (primary drainage) set to 0")
        elif getattr(f, "krPts", None) is not None:
            g = np.asarray(_field(f.krPts, "g"), dtype=float)
            sgtmax = g[regi - 1, 1]
            sgmax = g[regi - 1, 2]
            sgmin = g[reg - 1, 0]
        else:
            sgmax = np.ones(nregi)
            sgtmax = sgmax
            sgmin = np.zeros(nregi)
            warnings.warn("Max gas saturation (primary drainage) set to 1.")
            warnings.warn("Min gas saturation (primary drainage) set to 0")
        assert np.size(sgmax) == nregi
        self.sg_max = np.asarray(sgmax, dtype=float)
        self.sg_min = np.asarray(sgmin, dtype=float)
        self.sg_tmax = np.asarray(sgtmax, dtype=float)
        self.nreg_imbibition = nregi

    def evaluate_gas_rel_perm(self, model: Any, sg: Any, state: Any) -> Any:
        sgMax = model.get_props(state, "sgMax")
        sgMax = ad_max(sg, sgMax)
        return self.evaluate_phase_relative_permeability_with_hysteresis(model, "g", sg, sgMax)

    def evaluate_phase_relative_permeability_with_hysteresis(self, model: Any, phase: str,
                                                             s: Any, sMax: Any) -> Any:
        f = model.fluid
        fn = getattr(f, f"kr{phase.upper()}")
        if getattr(f, "krHyst", None) is None:
            regions = _rock_regions(model)["saturation"]
            return evaluate_function_cell_subset_reg(self, fn, regions, s)
        fn = list(fn) if isinstance(fn, (list, tuple)) else [fn]
        krG = list(f.krG) if isinstance(f.krG, (list, tuple)) else [f.krG]
        nregi = self.nreg_imbibition
        regions = np.asarray(_rock_regions(model)["imbibition"]).ravel()
        sample, is_ad = get_sample_ad(s, sMax)
        kr = np.zeros(regions.size)
        if is_ad:
            kr = self.AutoDiffBackend.convert_to_ad(kr, sample)
        offset = int(regions.min()) - 1
        for reg in range(1, np.unique(regions).size + 1):
            id_cells = regions == reg + offset
            is_regi = np.any(np.asarray(f.krHyst) == reg + offset)
            if is_regi:
                kri_vars = (self.sg_max[reg - 1], self.sg_min[reg - 1], krG[reg - 1],
                            krG[reg - 1 + nregi], self.sg_tmax[reg - 1])
                fni = (lambda sg, sgi, kv=kri_vars:
                       scan_rel_perm(sg, sgi, kv, self.minSat, f.ehystr)[0])
                kr[id_cells] = evaluate_function_cell_subset_reg(
                    self, fni, regions, s[id_cells], sMax[id_cells])
            else:
                kr[id_cells] = evaluate_function_cell_subset_reg(
                    self, fn[reg - 1], regions, s[id_cells])
        return kr

    evaluateGasRelPerm = evaluate_gas_rel_perm
    evaluatePhaseRelativePermeabilityWithHysteresis = \
        evaluate_phase_relative_permeability_with_hysteresis


def scan_rel_perm(sg: Any, sgi: Any, kriVars, minSat: float, opts):
    """MRST's `[kri, tol, minSat, sgt] = scanRelPerm(sg, sgi, kriVars, minSat, opts)`."""
    nci = np.size(value(sg))
    assert nci == np.size(value(sgi))
    tol = SCAN_TOLERANCE
    kri = np.full(nci, np.nan)
    sample, is_ad = get_sample_ad(sg, sgi)
    if is_ad:
        kri = _like(kri, sample)
    hyst_model = opts[1]
    sgmx, sgmn, krG, krGib, sgtmx = kriVars
    sgv_all, sgi_v = np.asarray(value(sg)), np.asarray(value(sgi))
    imb = (sgv_all + tol < sgi_v) & (sgi_v > minSat + sgmn)
    sgt = None
    if np.any(imb):
        kri[~imb] = krG(sg[~imb])
        sgiv = sgi[imb]
        sgv = sg[imb]
        if hyst_model == KILLOUGH:
            a = opts[3]
            A = 1 + a*(sgmx - sgiv)
            C = 1/(sgtmx - sgmn) - 1/(sgmx - sgmn)
            sgt = sgmn + (sgiv - sgmn)/(A + C*(sgiv - sgmn))
            snorm = sgtmx + (sgv - sgt)*(sgmx - sgtmx)/(sgiv - sgt)
            kri[imb] = krGib(snorm)*krG(sgiv)/krG(sgmx)
            is_above = (np.asarray(value(kri)) > np.asarray(value(krG(sgv_all)))) & imb
            if np.any(is_above):
                kri[is_above] = krG(sg[is_above])
        else:
            raise ValueError("Indicated hysteresis model in item 2 of EHYSTR keyword in "
                             "the .DATA input file is not supported.")
        kri_imb = np.asarray(value(kri))[imb]
        is_below = (np.asarray(value(sgv)) < np.asarray(value(sgt))) & (kri_imb != 0)
        assert not np.any(is_below)
        kriv = np.asarray(value(kri))
        assert np.all(~np.isnan(kriv))
        assert np.all(kriv >= 0)
        assert np.all(kriv <= value(krG(value(sgmx))))
    else:
        kri = krG(sg)
    return kri, tol, minSat, sgt


scanRelPerm = scan_rel_perm


def _rock_regions(model: Any) -> dict:
    regions = getattr(model.rock, "regions", None)
    return dict(regions) if regions else {}


def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _like(values: np.ndarray, sample: Any) -> Any:
    """`values` as an AD value with `sample`'s blocks and zero derivatives."""
    return 0*sample[np.zeros(values.size, dtype=np.int64)] + values \
        if np.size(value(sample)) else sample


__all__ = ["HystereticRelativePermeability", "scan_rel_perm", "scanRelPerm"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_killough_scanning_curve_lies_below_drainage():
        krG = lambda s: np.asarray(s, dtype=float)**2
        krGib = lambda s: 0.5*np.asarray(s, dtype=float)**2
        opts = [0.1, 2, 0.1, 0.1, "KR"]
        sg, sgi = np.array([0.55, 0.6]), np.array([0.6, 0.6])
        kri, _, _, sgt = scan_rel_perm(sg, sgi, (0.8, 0.0, krG, krGib, 0.6), 0.05, opts)
        assert kri[1] == krG(0.6), "at its maximum the cell is on the drainage curve"
        assert 0 <= kri[0] < krG(0.55), "below its maximum it is on the scanning curve"
        assert sgt is not None and 0 < sgt[0] < 0.55
        print("ok test_killough_scanning_curve_lies_below_drainage")

    def test_drainage_everywhere_returns_the_drainage_curve():
        krG = lambda s: np.asarray(s, dtype=float)**2
        kri, *_ = scan_rel_perm(np.array([0.2, 0.4]), np.array([0.2, 0.4]),
                                (0.8, 0.0, krG, krG, 0.6), 0.05, [0, 2, 0, 0, "KR"])
        np.testing.assert_allclose(kri, [0.04, 0.16])
        print("ok test_drainage_everywhere_returns_the_drainage_curve")

    def test_an_unsupported_hysteresis_model_is_refused():
        krG = lambda s: np.asarray(s, dtype=float)**2
        try:
            scan_rel_perm(np.array([0.1]), np.array([0.5]), (0.8, 0.0, krG, krG, 0.6),
                          0.05, [0.1, 0, 0.1, 0.1, "KR"])
        except ValueError:
            print("ok test_an_unsupported_hysteresis_model_is_refused")
            return
        raise AssertionError("EHYSTR model 0 must be refused, as in MRST")

    def test_region_one_only_is_hysteretic_with_krhyst_one():
        # MRST compares krHyst (= 1) with the imbibition region number:
        # imbibition region 2 takes its drainage curve unscaled.
        model = SimpleNamespace(
            fluid=SimpleNamespace(krG=[lambda s: s**2, lambda s: 0.5*s**2], krHyst=1,
                                  ehystr=[0.1, 2, 0.1, 0.1, "KR"]),
            rock=SimpleNamespace(regions={"imbibition": np.array([2, 2, 2])}))
        prop = HystereticRelativePermeability.__new__(HystereticRelativePermeability)
        prop.nreg_imbibition, prop.minSat = 1, 0.05
        prop.sg_max, prop.sg_min, prop.sg_tmax = np.array([1.0]), np.array([0.0]), np.array([1.0])
        kr = prop.evaluate_phase_relative_permeability_with_hysteresis(
            model, "g", np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.5, 0.5]))
        np.testing.assert_allclose(kr, [0.01, 0.04, 0.09])
        print("ok test_region_one_only_is_hysteretic_with_krhyst_one")

    test_killough_scanning_curve_lies_below_drainage()
    test_drainage_everywhere_returns_the_drainage_curve()
    test_an_unsupported_hysteresis_model_is_refused()
    test_region_one_only_is_hysteretic_with_krhyst_one()
    print("\nAll tests passed!")
