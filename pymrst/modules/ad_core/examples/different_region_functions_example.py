"""Regional relative-permeability workflow from MRST's AD-core examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.adi import value


@dataclass(frozen=True)
class RegionalRelPermResult:
    regions: np.ndarray
    water: np.ndarray
    oil: np.ndarray
    water_jacobian: np.ndarray
    oil_jacobian: np.ndarray


def _evaluate(model, saturation) -> RegionalRelPermResult:
    state = State(
        pressure=np.ones(model.G.cells.num),
        sW=np.asarray(saturation, dtype=float),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    state_ad = model.init_state_ad(state)
    relperm = model.get_props(state_ad, "RelativePermeability")
    function = model.FlowPropertyFunctions.get_state_function(
        "RelativePermeability"
    )
    return RegionalRelPermResult(
        regions=np.asarray(function.regions).copy(),
        water=np.asarray(value(relperm[0])),
        oil=np.asarray(value(relperm[1])),
        water_jacobian=relperm[0].jac[1].toarray(),
        oil_jacobian=relperm[1].jac[1].toarray(),
    )


def different_region_functions_example(
    saturation=(0.2, 0.4, 0.6, 0.8),
    regions=(1, 1, 2, 2),
    *,
    via_rock: bool = True,
) -> RegionalRelPermResult:
    """Evaluate quadratic region 1 and linear region 2 relperm curves.

    Set ``via_rock=False`` to exercise the direct state-function
    ``set_regions`` approach shown in the second half of the MRST example.
    """

    saturation = np.asarray(saturation, dtype=float).ravel()
    regions = np.asarray(regions, dtype=np.int64).ravel()
    if saturation.size != regions.size:
        raise ValueError("saturation and regions must have equal lengths")
    grid = compute_geometry(
        cart_grid([saturation.size, 1, 1], [float(saturation.size), 1.0, 1.0])
    )
    rock = make_rock(grid, 1e-13, 0.3)
    if via_rock:
        rock.regions = {"saturation": regions}
    fluid = init_simple_adi_fluid(
        mu=[1.0, 1.0], rho=[100.0, 100.0], n=[1.0, 1.0]
    )
    curves = [lambda s: s**2, lambda s: s]
    fluid.kr_w = curves
    fluid.kr_o = curves
    model = GenericBlackOilModelAD(grid, rock, fluid, [])
    if not via_rock:
        model.FlowPropertyFunctions.get_state_function(
            "RelativePermeability"
        ).set_regions(regions)
    return _evaluate(model, saturation)


differentRegionFunctionsExample = different_region_functions_example

__all__ = [
    "RegionalRelPermResult",
    "different_region_functions_example",
    "differentRegionFunctionsExample",
]
