"""MRST ``ad-core/statefunctions/flowprops/virtual/SaturationProperty.m``.

Mixin for properties evaluated as functions of saturation: connate water,
critical saturations, and ECLIPSE end-point scaling (two- or three-point) of
the saturation tables per cell.

End points follow MRST's `krPts` layout - one row per saturation region with
columns connate (`L`), critical (`CR`), maximum (`U`) and maximum relative
permeability (`KM`) - indexed here from 0. Region maps stay MRST's 1-based
SATNUM. `rock.krscale.<type>.<phase>` holds the per-cell scaled points in the
same four columns, NaN where the table value applies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from pymrst.utils.adi import value

WHOLE: str = ":"
CONNATE, CRITICAL, MAXIMUM, KR_MAXIMUM = 0, 1, 2, 3


class SaturationProperty:
    """End-point handling shared by relative permeability and capillarity."""

    scalingActive = False
    swcon = None
    scalers = None
    cell_subset: Any = WHOLE

    def store_connate_water(self, model: Any, *args: Any):
        self.swcon = self.get_connate_water(model, *args)
        return self

    def store_scalers(self, model: Any, *args: Any):
        krscale = _field(model.rock, "krscale")
        if krscale is not None:
            out: Dict[str, Dict[str, Any]] = {}
            pts_names = set(_names(model.fluid.krPts))
            for scale_type in _names(krscale):
                phases = [n for n in _names(_field(krscale, scale_type))
                          if n in pts_names]
                out[scale_type] = {phase: self.get_scalers(model, phase, WHOLE,
                                                           scale_type)
                                   for phase in phases}
            self.scalers = out
        return self

    def get_connate_water(self, model: Any, state: Any = None) -> Any:
        if self.swcon is not None:
            return self.swcon
        krscale = _field(model.rock, "krscale")
        drainage = _field(krscale, "drainage")
        if self.scalingActive and krscale is not None and \
                _field(drainage, "w") is not None:
            swcon = np.array(_rows(np.asarray(drainage["w"] if isinstance(drainage, dict)
                                              else drainage.w),
                                   self.cell_subset)[:, CONNATE], dtype=float)
            missing = np.isnan(swcon)
            if np.any(missing):
                table = np.asarray(model.fluid.krPts["w"])
                regions = np.asarray(self.regions)
                swcon[missing] = table[regions[missing] - 1, CONNATE]
            return swcon
        pts = getattr(model.fluid, "krPts", None)
        if pts and "w" in pts:
            table = np.asarray(pts["w"])
            return table[np.asarray(self.regions).ravel() - 1, CONNATE]
        return np.zeros(model.G.cells.num)

    def get_critical_phase_saturations(self, model: Any, state: Any,
                                       drainage: bool = True) -> np.ndarray:
        """`1 x nph` critical saturations of one saturation region.

        MRST fills `s_min(i)` per phase; with a per-cell region map that
        assignment raises in MATLAB, so MRST supports one region here and so
        does this.
        """
        satnum = self.regions_imbibition if drainage else self.regions
        satnum = 1 if satnum is None or np.size(satnum) == 0 else satnum
        if np.size(satnum) > 1:
            raise ValueError(
                "critical saturations are defined for a single saturation "
                "region; MRST's getCriticalPhaseSaturations fails the same way "
                "for a per-cell region map")
        region = int(np.ravel(satnum)[0]) - 1
        phases = model.get_phase_names()
        s_min = np.zeros((1, len(phases)))
        pts = getattr(model.fluid, "krPts", None)
        if pts:
            for i, letter in enumerate(phases):
                ph = letter.lower()
                if ph == "o" and "o" not in pts:
                    if model.water and model.gas:
                        s_min[0, i] = max(pts["ow"][region, CRITICAL],
                                          pts["og"][region, CRITICAL])
                    elif model.water:
                        s_min[0, i] = pts["ow"][region, CRITICAL]
                    elif model.gas:
                        s_min[0, i] = pts["og"][region, CRITICAL]
                else:
                    s_min[0, i] = pts[ph][region, CRITICAL]
        return s_min

    def get_scalers(self, model: Any, phase: str, cells: Any = WHOLE,
                    scale_type: str = "drainage") -> Dict[str, Any]:
        if self.scalers is None:
            pts = _field(model.rock.krscale, scale_type)
            regions = _rows(np.asarray(self.regions), cells)
            npts = self.relpermPoints
            f = model.fluid
            if npts == 2:
                m, c, p, k = self.get_two_point_scalers(pts, phase, regions, f, cells)
            elif npts == 3:
                m, c, p, k = self.get_three_point_scalers(pts, phase, regions, f, cells)
            else:
                raise ValueError("Unknown number of scaling points")
            return {"m": m, "c": c, "p": p, "k": k, "k_max_m": k[1]/k[0],
                    "points": npts}
        scaler = dict(self.scalers[scale_type][phase])
        if not (isinstance(cells, str) and cells == WHOLE):
            for key in ("m", "c", "p", "k"):
                item = scaler[key]
                scaler[key] = ([_rows(x, cells) for x in item]
                               if isinstance(item, list) else _rows(item, cells))
            scaler["k_max_m"] = _rows(scaler["k_max_m"], cells)
        return scaler

    @staticmethod
    def get_two_point_scalers(pts, ph, reg, f, cells):
        get = SaturationProperty.get_sat_point_picker(f, pts, reg, cells)
        if ph in ("w", "g"):
            su, SU = get(ph, MAXIMUM)
        elif ph in ("ow", "og"):
            swl, SWL = get("w", CONNATE) if "w" in f.krPts else (0.0, 0.0)
            sgl, SGL = get("g", CONNATE) if "g" in f.krPts else (0.0, 0.0)
            SU = 1 - SWL - SGL
            su = 1 - swl - sgl
        else:
            raise ValueError(f"No valid scalers for phase {ph}")
        scr, SCR = get(ph, CRITICAL)
        m = (su - scr)/(SU - SCR)
        c = scr - SCR*m
        k = list(get(ph, KR_MAXIMUM))
        return m, c, [SCR, SU], k

    @staticmethod
    def get_three_point_scalers(pts, ph, reg, f, cells):
        get = SaturationProperty.get_sat_point_picker(f, pts, reg, cells)
        if ph == "w":
            sowcr, SOWCR = get("ow", CRITICAL)
            sgl, SGL = get("g", CONNATE)
            SR, sr = 1 - SOWCR - SGL, 1 - sowcr - sgl
            su, SU = get("w", MAXIMUM)
        elif ph == "g":
            sogcr, SOGCR = get("og", CRITICAL)
            swl, SWL = get("w", CONNATE)
            SR, sr = 1 - SOGCR - SWL, 1 - sogcr - swl
            su, SU = get("g", MAXIMUM)
        elif ph == "ow":
            swcr, SWCR = get("w", CRITICAL)
            sgl, SGL = get("g", CONNATE)
            SR, sr = 1 - SWCR - SGL, 1 - swcr - sgl
            swl, SWL = get("w", CONNATE)
            SU, su = 1 - SWL - SGL, 1 - swl - sgl
        elif ph == "og":
            sgcr, SGCR = get("g", CRITICAL)
            swl, SWL = get("w", CONNATE)
            sgl, SGL = get("g", CONNATE)
            SR, sr = 1 - SGCR - SWL, 1 - sgcr - swl
            SU, su = 1 - SGL - SWL, 1 - sgl - swl
        else:
            raise ValueError(f"No valid scalers for phase {ph}")
        scr, SCR = get(ph, CRITICAL)
        m1 = (sr - scr)/(SR - SCR)
        c1 = scr - SCR*m1
        m2 = np.array((su - sr)/(SU - SR), dtype=float, copy=True)
        c2 = np.array(sr - SR*m2, dtype=float, copy=True)
        flat = np.asarray(SU <= SR)
        if np.any(flat):
            m2[flat] = 0.0
            c2[flat] = 0.0
        k = list(get(ph, KR_MAXIMUM))
        return [m1, m2], [c1, c2], [SCR, SR, SU], k

    @staticmethod
    def get_sat_point_picker(f: Any, pts: Any, reg: np.ndarray,
                             cells: Any) -> Callable[[str, int], Tuple[Any, Any]]:
        """`get(phase, column)` -> `(table value, scaled value)` per cell."""
        def table(phase: str, index: int):
            return np.asarray(f.krPts[phase])[np.asarray(reg).ravel() - 1, index]

        def scaled(phase: str, index: int):
            return _rows(np.asarray(_field(pts, phase)), cells)[:, index]

        return lambda phase, index: SaturationProperty.get_pair(phase, index,
                                                               table, scaled)

    @staticmethod
    def get_pair(phase: str, index: int, fn1, fn2):
        v1 = np.asarray(fn1(phase, index), dtype=float)
        v2 = np.array(fn2(phase, index), dtype=float, copy=True)
        missing = np.isnan(v2)
        v2[missing] = np.broadcast_to(v1, v2.shape)[missing]
        return v1, v2

    storeConnateWater = store_connate_water
    storeScalers = store_scalers
    getConnateWater = get_connate_water
    getCriticalPhaseSaturations = get_critical_phase_saturations
    getScalers = get_scalers
    getTwoPointScalers = get_two_point_scalers
    getThreePointScalers = get_three_point_scalers
    getSatPointPicker = get_sat_point_picker
    getPair = get_pair


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _names(obj: Any):
    if isinstance(obj, dict):
        return list(obj)
    return [k for k in vars(obj) if not k.startswith("_")]


def _rows(array: Any, cells: Any) -> Any:
    if isinstance(cells, str) and cells == WHOLE:
        return array
    return np.asarray(array)[cells]


__all__ = ["SaturationProperty"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(pts, krscale=None):
        return SimpleNamespace(fluid=SimpleNamespace(krPts=pts),
                               rock=SimpleNamespace(krscale=krscale),
                               G=SimpleNamespace(cells=SimpleNamespace(num=2)),
                               water=True, gas=False,
                               get_phase_names=lambda: "WO")

    class _Prop(SaturationProperty):
        regions = np.array([1, 1])
        regions_imbibition = None
        relpermPoints = 2

    PTS = {"w": np.array([[0.2, 0.25, 1.0, 0.9]]),
           "ow": np.array([[0.0, 0.3, 0.8, 1.0]])}

    def test_connate_water_comes_from_the_region_table():
        np.testing.assert_allclose(_Prop().get_connate_water(_model(PTS)), [0.2, 0.2])
        print("ok test_connate_water_comes_from_the_region_table")

    def test_critical_saturations_follow_the_oil_water_table_for_oil():
        s_min = _Prop().get_critical_phase_saturations(_model(PTS), None)
        np.testing.assert_allclose(s_min, [[0.25, 0.3]])
        print("ok test_critical_saturations_follow_the_oil_water_table_for_oil")

    def test_two_point_scaling_maps_the_scaled_range_onto_the_table():
        scaled = {"w": np.array([[np.nan, 0.35, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan]])}
        model = _model(PTS, SimpleNamespace(drainage=scaled))
        scaler = _Prop().get_scalers(model, "w")
        # Cell 0: critical moved 0.35 -> 0.25, maximum stays 1.0.
        np.testing.assert_allclose(scaler["m"][0], (1.0 - 0.25)/(1.0 - 0.35))
        np.testing.assert_allclose(scaler["m"][1], 1.0)
        print("ok test_two_point_scaling_maps_the_scaled_range_onto_the_table")

    test_connate_water_comes_from_the_region_table()
    test_critical_saturations_follow_the_oil_water_table_for_oil()
    test_two_point_scaling_maps_the_scaled_range_onto_the_table()
    print("\nAll tests passed!")
