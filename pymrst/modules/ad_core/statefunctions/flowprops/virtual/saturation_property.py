"""MRST saturation-property endpoint-scaling utilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pymrst.utils.adi import value


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _region_indices(regions: Any, cells: np.ndarray, count: int) -> np.ndarray:
    if regions is None:
        return np.zeros(cells.size, dtype=int)
    region = np.asarray(regions, dtype=int).ravel()
    if region.size == 1:
        region = np.full(count, region.item(), dtype=int)
    region = region[cells]
    # StateFunction region maps use MRST's one-based convention.
    if region.size and np.min(region) >= 1:
        region = region - 1
    return region


class SaturationProperty:
    """Virtual base for saturation functions and ECLIPSE endpoint scaling."""

    def __init__(self, model: Any = None):
        self._swcon: Optional[np.ndarray] = None
        self.scalers: dict[str, dict[str, dict[str, Any]]] = {}
        self.scaling_active = bool(
            model is not None and _field(model.rock, "krscale") is not None
        )
        self.cell_subset: Any = None
        self.regions = getattr(self, "regions", None)
        self.regions_imbibition = getattr(self, "regions_imbibition", None)
        self.relperm_points = getattr(self, "relperm_points", 2)

    def _raw_fluid(self, model: Any) -> Any:
        return getattr(model, "raw_fluid", model.fluid)

    def _cells(self, model: Any, cells: Any = None) -> np.ndarray:
        selected = self.cell_subset if cells is None else cells
        if selected is None or isinstance(selected, str):
            return np.arange(model.G.cells.num, dtype=int)
        return np.asarray(selected, dtype=int).ravel()

    # -- MRST SaturationProperty.m endpoint scaling, ported ----------------
    #
    # A relative-permeability table is measured on one rock. Endpoint scaling
    # is how that table is stretched onto each cell's own saturation limits,
    # so a cell with a higher connate water does not start mobilising water at
    # the table's connate value.
    #
    # The saturation points are indexed the way MRST indexes them - and they
    # are one-based there, so the constants are named rather than written as
    # literals at every use.

    #: connate saturation, always present
    L = 1
    #: critical saturation - the first point at which the phase is mobile
    CR = 2
    #: saturation at which the maximum relative permeability occurs
    U = 3
    #: the maximum relative permeability itself
    KM = 4

    @staticmethod
    def get_pair(phase: str, index: int, table_fn, scaled_fn):
        """`(table value, scaled value)` for one saturation point.

        The scaled value falls back to the table's wherever it is NaN, and
        that fallback is the load-bearing part: a deck supplies endpoints for
        some cells and not others, and a cell with no override must inherit
        the table rather than get a NaN that propagates into every
        permeability it touches.
        """
        table = np.asarray(table_fn(phase, index), dtype=float)
        scaled = np.array(scaled_fn(phase, index), dtype=float, copy=True)
        missing = np.isnan(scaled)
        scaled[missing] = np.broadcast_to(table, scaled.shape)[missing]
        return table, scaled

    getPair = get_pair

    @classmethod
    def get_sat_point_picker(cls, fluid: Any, points: Any, regions, cells):
        """A getter returning `(table, scaled)` for any `(phase, index)`.

        MRST returns the index constants alongside so a caller can write
        `get('w', L)`; here they are class attributes, so only the getter is
        returned.
        """
        kr_points = _field(fluid, "krPts", "kr_points")

        def table_fn(phase, index):
            column = np.asarray(_field(kr_points, phase), dtype=float)
            return np.atleast_2d(column)[regions, index - 1]

        def scaled_fn(phase, index):
            column = _field(points, phase)
            if column is None:
                return np.full(np.size(cells), np.nan)
            return np.atleast_2d(np.asarray(column, dtype=float))[cells,
                                                                  index - 1]

        return lambda phase, index: cls.get_pair(phase, index, table_fn,
                                                  scaled_fn)

    getSatPointPicker = get_sat_point_picker

    @classmethod
    def get_two_point_scalers(cls, points, phase: str, regions, fluid, cells):
        """A single linear map taking scaled saturation onto table saturation.

        `s = m*S + c`, fixed by matching the critical point and the upper
        point. Two points, one segment - which is why the maximum
        relative-permeability location is preserved but nothing between the
        endpoints is controlled.
        """
        get = cls.get_sat_point_picker(fluid, points, regions, cells)
        kr_points = _field(fluid, "krPts", "kr_points")

        if phase in ("w", "g"):
            su, SU = get(phase, cls.U)
        elif phase in ("ow", "og"):
            swl, SWL = ((0.0, 0.0) if _field(kr_points, "w") is None
                        else get("w", cls.L))
            sgl, SGL = ((0.0, 0.0) if _field(kr_points, "g") is None
                        else get("g", cls.L))
            su, SU = 1.0 - swl - sgl, 1.0 - SWL - SGL
        else:
            raise ValueError(f"no two-point scalers for phase {phase!r}")

        scr, SCR = get(phase, cls.CR)
        m = (su - scr)/(SU - SCR)
        c = scr - SCR*m
        return m, c, [SCR, SU], list(get(phase, cls.KM))

    getTwoPointScalers = get_two_point_scalers

    @classmethod
    def get_three_point_scalers(cls, points, phase: str, regions, fluid, cells):
        """Two linear segments joined at the residual saturation.

        Three-point scaling pins the *displacing* phase's residual as well as
        the endpoints, so the mobile range is mapped correctly rather than only
        its ends. The middle point is what two-point scaling cannot represent.

        The `SU <= SR` case zeroes the upper segment. That interval is
        degenerate - there is no room between the residual and the maximum -
        and scaling across it would divide by zero.
        """
        get = cls.get_sat_point_picker(fluid, points, regions, cells)

        if phase == "w":
            sowcr, SOWCR = get("ow", cls.CR)
            sgl, SGL = get("g", cls.L)
            sr, SR = 1.0 - sowcr - sgl, 1.0 - SOWCR - SGL
            su, SU = get("w", cls.U)
        elif phase == "g":
            sogcr, SOGCR = get("og", cls.CR)
            swl, SWL = get("w", cls.L)
            sr, SR = 1.0 - sogcr - swl, 1.0 - SOGCR - SWL
            su, SU = get("g", cls.U)
        elif phase == "ow":
            swcr, SWCR = get("w", cls.CR)
            sgl, SGL = get("g", cls.L)
            sr, SR = 1.0 - swcr - sgl, 1.0 - SWCR - SGL
            swl, SWL = get("w", cls.L)
            su, SU = 1.0 - swl - sgl, 1.0 - SWL - SGL
        elif phase == "og":
            sgcr, SGCR = get("g", cls.CR)
            swl, SWL = get("w", cls.L)
            sgl, SGL = get("g", cls.L)
            sr, SR = 1.0 - sgcr - swl, 1.0 - SGCR - SWL
            su, SU = 1.0 - sgl - swl, 1.0 - SGL - SWL
        else:
            raise ValueError(f"no three-point scalers for phase {phase!r}")

        scr, SCR = get(phase, cls.CR)
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = (sr - scr)/(SR - SCR)
            m2 = (su - sr)/(SU - SR)
        c1 = scr - SCR*m1
        c2 = sr - SR*m2

        degenerate = np.asarray(SU) <= np.asarray(SR)
        m2 = np.where(degenerate, 0.0, m2)
        c2 = np.where(degenerate, 0.0, c2)
        return ([m1, m2], [c1, c2], [SCR, SR, SU],
                list(get(phase, cls.KM)))

    getThreePointScalers = get_three_point_scalers

    def store_connate_water(self, model: Any) -> "SaturationProperty":
        self._swcon = self.get_connate_water(model)
        return self

    def store_scalers(self, model: Any) -> "SaturationProperty":
        krscale = _field(model.rock, "krscale")
        fluid_points = _field(self._raw_fluid(model), "krPts")
        if krscale is None or fluid_points is None:
            return self
        stored: dict[str, dict[str, dict[str, Any]]] = {}
        items = krscale.items() if isinstance(krscale, dict) else vars(krscale).items()
        for scaling_type, phase_points in items:
            phase_items = (
                phase_points.items()
                if isinstance(phase_points, dict)
                else vars(phase_points).items()
            )
            available = {}
            for phase, _ in phase_items:
                if _field(fluid_points, phase) is not None:
                    available[phase] = self.get_scalers(
                        model, phase, type=scaling_type
                    )
            stored[scaling_type] = available
        self.scalers = stored
        self.scaling_active = bool(stored)
        return self

    def get_connate_water(self, model: Any, state: Any = None) -> np.ndarray:
        if self._swcon is not None:
            return self._swcon
        cells = self._cells(model)
        raw = self._raw_fluid(model)
        krscale = _field(model.rock, "krscale")
        drainage = _field(krscale, "drainage") if krscale is not None else None
        scaled_water = _field(drainage, "w") if drainage is not None else None
        fluid_points = _field(raw, "krPts")
        table_water = (
            _field(fluid_points, "w") if fluid_points is not None else None
        )
        regions = _region_indices(self.regions, cells, model.G.cells.num)

        if self.scaling_active and scaled_water is not None:
            connate = np.asarray(scaled_water, dtype=float)[cells, 0].copy()
            missing = np.isnan(value(connate))
            if np.any(missing) and table_water is not None:
                connate[missing] = np.asarray(table_water, dtype=float)[
                    regions[missing], 0
                ]
            return connate
        if table_water is not None:
            return np.asarray(table_water, dtype=float)[regions, 0]
        swcon = float(getattr(raw, "swcon", 0.0) or 0.0)
        return np.full(cells.size, swcon)

    def get_critical_phase_saturations(
        self, model: Any, state: Any = None, drainage: bool = True
    ) -> np.ndarray:
        raw = self._raw_fluid(model)
        points = _field(raw, "krPts")
        nph = model.n_phases
        if points is None:
            return np.zeros(nph)
        regions = self.regions if drainage else self.regions_imbibition
        if regions is None:
            region = 0
        else:
            region = int(np.asarray(regions).ravel()[0])
            if region >= 1:
                region -= 1
        phases = [str(name).lower() for name in model.get_phase_names()]
        output = np.zeros(nph)
        for index, phase in enumerate(phases):
            if phase == "o" and _field(points, "o") is None:
                oil_water = _field(points, "ow")
                oil_gas = _field(points, "og")
                candidates = []
                if oil_water is not None:
                    candidates.append(np.asarray(oil_water)[region, 1])
                if oil_gas is not None:
                    candidates.append(np.asarray(oil_gas)[region, 1])
                output[index] = max(candidates) if candidates else 0.0
            else:
                phase_points = _field(points, phase)
                if phase_points is not None:
                    output[index] = np.asarray(phase_points)[region, 1]
        return output

    def get_scalers(
        self, model: Any, phase: str, cells: Any = None,
        type: str = "drainage",
    ) -> dict[str, Any]:
        selected = self._cells(model, cells)
        if self.scalers:
            scaler = self.scalers[type][phase]
            if cells is None or isinstance(cells, str):
                return scaler
            output = dict(scaler)
            for name in ("m", "c", "p", "k"):
                values = scaler[name]
                output[name] = [
                    np.asarray(item)[selected] for item in values
                ]
            output["k_max_m"] = np.asarray(scaler["k_max_m"])[selected]
            return output

        krscale = _field(model.rock, "krscale")
        scaling_points = _field(_field(krscale, type), phase)
        if scaling_points is None:
            raise KeyError(f"missing {type} endpoint scalers for phase {phase}")
        raw = self._raw_fluid(model)
        fluid_points = _field(raw, "krPts")
        regions = _region_indices(self.regions, selected, model.G.cells.num)
        getter = self._point_getter(
            fluid_points, _field(krscale, type), regions, selected
        )
        if self.relperm_points == 2:
            m, c, p, k = self._two_point_scalers(
                phase, getter, fluid_points
            )
        elif self.relperm_points == 3:
            m, c, p, k = self._three_point_scalers(phase, getter)
        else:
            raise ValueError(
                f"unknown number of scaling points: {self.relperm_points}"
            )
        return {
            "m": m, "c": c, "p": p, "k": k,
            "k_max_m": k[1] / k[0],
            "points": self.relperm_points,
        }

    @staticmethod
    def _point_getter(fluid_points, scaling_points, regions, cells):
        def get(phase: str, column: int):
            table = np.asarray(_field(fluid_points, phase), dtype=float)[
                regions, column
            ]
            scaled = np.asarray(_field(scaling_points, phase), dtype=float)[
                cells, column
            ].copy()
            missing = np.isnan(value(scaled))
            scaled[missing] = table[missing]
            return table, scaled
        return get

    @staticmethod
    def _two_point_scalers(phase, get, fluid_points):
        lower, critical, upper, maximum = 0, 1, 2, 3
        if phase in ("w", "g"):
            su, SU = get(phase, upper)
        elif phase in ("ow", "og"):
            if _field(fluid_points, "w") is not None:
                swl, SWL = get("w", lower)
            else:
                swl = SWL = 0.0
            if _field(fluid_points, "g") is not None:
                sgl, SGL = get("g", lower)
            else:
                sgl = SGL = 0.0
            SU, su = 1.0 - SWL - SGL, 1.0 - swl - sgl
        else:
            raise ValueError(f"invalid phase {phase!r}")
        scr, SCR = get(phase, critical)
        m = [(su - scr) / (SU - SCR)]
        c = [scr - SCR * m[0]]
        p = [SCR, SU]
        k = list(get(phase, maximum))
        return m, c, p, k

    @staticmethod
    def _three_point_scalers(phase, get):
        lower, critical, upper, maximum = 0, 1, 2, 3
        if phase == "w":
            sowcr, SOWCR = get("ow", critical)
            sgl, SGL = get("g", lower)
            SR, sr = 1.0 - SOWCR - SGL, 1.0 - sowcr - sgl
            su, SU = get("w", upper)
        elif phase == "g":
            sogcr, SOGCR = get("og", critical)
            swl, SWL = get("w", lower)
            SR, sr = 1.0 - SOGCR - SWL, 1.0 - sogcr - swl
            su, SU = get("g", upper)
        elif phase == "ow":
            swcr, SWCR = get("w", critical)
            sgl, SGL = get("g", lower)
            SR, sr = 1.0 - SWCR - SGL, 1.0 - swcr - sgl
            swl, SWL = get("w", lower)
            SU, su = 1.0 - SWL - SGL, 1.0 - swl - sgl
        elif phase == "og":
            sgcr, SGCR = get("g", critical)
            swl, SWL = get("w", lower)
            sgl, SGL = get("g", lower)
            SR, sr = 1.0 - SGCR - SWL, 1.0 - sgcr - swl
            SU, su = 1.0 - SGL - SWL, 1.0 - sgl - swl
        else:
            raise ValueError(f"invalid phase {phase!r}")
        scr, SCR = get(phase, critical)
        m1 = (sr - scr) / (SR - SCR)
        c1 = scr - SCR * m1
        with np.errstate(divide="ignore", invalid="ignore"):
            m2 = (su - sr) / (SU - SR)
        c2 = sr - SR * m2
        invalid = SU <= SR
        m2 = np.where(invalid, 0.0, m2)
        c2 = np.where(invalid, 0.0, c2)
        k = list(get(phase, maximum))
        return [m1, m2], [c1, c2], [SCR, SR, SU], k

    storeConnateWater = store_connate_water
    storeScalers = store_scalers
    getConnateWater = get_connate_water
    getCriticalPhaseSaturations = get_critical_phase_saturations
    getScalers = get_scalers


__all__ = ["SaturationProperty"]
