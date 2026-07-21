"""SaturationProperty (MRST ad-core/statefunctions/flowprops/virtual).

Ports the reusable part of MRST's `SaturationProperty.m`: the virtual base for
saturation-function state functions (relative permeability, capillary
pressure), holding the connate-water saturation and the endpoint-scaling flag.

MRST's full class also carries the ECLIPSE endpoint-scaling machinery
(``rock.krscale`` / ``fluid.krPts`` - SWL/SWCR/SWU-style scalers). That path is
not ported here because PyMRST's rock does not carry `krscale`; the flag
`scaling_active` is therefore False and `store_scalers` is a no-op. What is
ported is what the relperm hierarchy needs with unscaled saturation functions:
the connate water, read from the fluid's `swcon`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class SaturationProperty:
    """Virtual base for saturation-function state functions.

    Mixed into `BaseRelativePermeability` (and, later, capillary pressure).
    Carries the connate water and the (currently inactive) endpoint-scaling
    flag, matching MRST's property surface so a scaled variant can be added
    without changing the relperm classes.
    """

    scaling_active: bool = False

    def __init__(self, model: Any = None):
        self._swcon: Optional[np.ndarray] = None

    def store_connate_water(self, model: Any) -> "SaturationProperty":
        self._swcon = self.get_connate_water(model)
        return self

    def store_scalers(self, model: Any) -> "SaturationProperty":
        # Endpoint scaling is not ported (no rock.krscale in PyMRST).
        return self

    def get_connate_water(self, model: Any, state: Any = None) -> np.ndarray:
        """Connate water saturation per cell, from the fluid's ``swcon``.

        Reads the raw fluid (`model.raw_fluid`), which carries `swcon`; the
        `PhaseFluid` adapter on `model.fluid` does not expose it.
        """
        if self._swcon is not None:
            return self._swcon
        raw = getattr(model, "raw_fluid", model.fluid)
        swcon = float(getattr(raw, "swcon", 0.0) or 0.0)
        return np.full(model.G.cells.num, swcon)


__all__ = ["SaturationProperty"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model(swcon=0.0):
        g = compute_geometry(cart_grid([3, 3, 1], [30.0, 30.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        fluid.swcon = swcon
        # raw fluid carries swcon; expose on the raw object
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        return GenericBlackOilModel(g, rock, fluid, w)

    def test_connate_water_from_fluid():
        model = _model(swcon=0.12)
        sp = SaturationProperty(model)
        swcon = sp.get_connate_water(model)
        assert swcon.shape == (model.G.cells.num,)
        assert np.allclose(swcon, 0.12)
        assert sp.scaling_active is False
        print("ok test_connate_water_from_fluid")

    def test_stored_swcon_is_returned():
        model = _model(swcon=0.2)
        sp = SaturationProperty(model).store_connate_water(model)
        assert np.allclose(sp.get_connate_water(model), 0.2)
        print("ok test_stored_swcon_is_returned")

    test_connate_water_from_fluid()
    test_stored_swcon_is_returned()
    print("\nAll tests passed!")
