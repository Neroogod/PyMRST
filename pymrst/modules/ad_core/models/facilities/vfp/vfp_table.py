"""MRST ``ad-core/models/facilities/vfp/VFPTable.m``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class VFPTable:
    """Tabulated BHP as a function of flow, THP, and optional ratios."""

    flow: np.ndarray
    thp: np.ndarray
    parameter_values: list[np.ndarray] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)
    bhp_table: np.ndarray | None = None
    flow_type: str = ""
    ref_depth: float = 0.0
    gas_ratio_type: str = ""
    water_ratio_type: str = ""
    _interp: RegularGridInterpolator | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.flow = np.asarray(self.flow, dtype=float)
        self.thp = np.asarray(self.thp, dtype=float)
        self.parameter_values = [np.asarray(v, dtype=float) for v in self.parameter_values]
        if self.bhp_table is not None:
            self.bhp_table = np.asarray(self.bhp_table, dtype=float)
            self.update_interpolants()

    @classmethod
    def from_deck(cls, deck_table: Mapping) -> "VFPTable":
        """Construct from a converted ``VFPPROD`` or ``VFPINJ`` deck record."""
        if "Q" in deck_table:
            pval = [deck_table["WFR"], deck_table["GFR"]]
            pname = ["WFR", "GFR"]
            table = np.asarray(deck_table["Q"], dtype=float)
            alq = np.asarray(deck_table.get("ALQ", []), dtype=float)
            if alq.size > 1 and deck_table.get("ALQID"):
                pval.append(alq)
                pname.append("ALQ")
            elif table.ndim == 5:
                table = table[..., 0]
            return cls(
                deck_table["FLO"],
                deck_table["THP"],
                pval,
                pname,
                table,
                flow_type=str(deck_table.get("FLOID", "")).lower(),
                ref_depth=float(deck_table.get("depth", deck_table.get("DEPTH", 0.0))),
                water_ratio_type=str(deck_table.get("WFRID", "")).lower(),
                gas_ratio_type=str(deck_table.get("GFRID", "")).lower(),
            )
        return cls(
            deck_table["FLO"],
            deck_table["THP"],
            [],
            [],
            deck_table["BHP"],
            flow_type=str(deck_table.get("FLOID", "")).lower(),
            ref_depth=float(deck_table.get("depth", deck_table.get("DEPTH", 0.0))),
        )

    def update_interpolants(self):
        axes = [self.flow, self.thp, *self.parameter_values]
        self._interp = RegularGridInterpolator(
            axes,
            self.bhp_table,
            bounds_error=False,
            fill_value=None,
        )
        return self

    def evaluate_bhp(self, flow, thp, *params):
        """Evaluate tabulated bottom-hole pressure."""
        if self._interp is None:
            raise ValueError("VFPTable has no BHP table")
        arrays = np.broadcast_arrays(flow, thp, *params)
        pts = np.column_stack([np.asarray(a, dtype=float).ravel() for a in arrays])
        out = self._interp(pts).reshape(arrays[0].shape)
        return out


__all__ = ["VFPTable"]


if __name__ == "__main__":
    def test_vfp_table_interpolates():
        flo = np.array([0.0, 10.0])
        thp = np.array([100.0, 200.0])
        wfr = np.array([0.0, 1.0])
        t = np.zeros((2, 2, 2))
        for i, f in enumerate(flo):
            for j, p in enumerate(thp):
                for k, w in enumerate(wfr):
                    t[i, j, k] = p + 2.0 * f + 5.0 * w
        vfp = VFPTable(flo, thp, [wfr], ["WFR"], t, flow_type="liquid")
        got = vfp.evaluate_bhp(5.0, 150.0, 0.5)
        assert np.isclose(got, 150.0 + 10.0 + 2.5)
        print("ok test_vfp_table_interpolates")

    test_vfp_table_interpolates()
