"""Extended shale fluid registry (MRST ``utils/getShaleCompFluidCase.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.shale.compositional_fns.get_shale_comp_fluid_case import (
    _info,
    _mixture,
    _table_case,
    get_shale_comp_fluid_case as get_main_shale_comp_fluid_case,
)
from pymrst.modules.shale.compositional_fns.sorbed_compositional_fluid import (
    SorbedCompositionalFluid,
)
from pymrst.utils.units import PSIA, RANKINE


def get_shale_comp_fluid_case(name: str):
    """Return any case from MRST's extended utility fluid catalog.

    The utility owner adds seven SlotDrill/natural-variable cases to the
    complete 18-case compositional registry.
    """
    key = name.lower().replace("-", "_")
    if key in {"oil_1", "oil_1_modified", "oil_1_modified_2"}:
        if key == "oil_1":
            names = ("Methane", "CarbonDioxide", "n-Decane")
            injection, initial = [1, 0, 0], [.25, .25, .5]
        elif key == "oil_1_modified":
            names = ("Methane", "CarbonDioxide", "n-Decane", "Nitrogen")
            injection, initial = [0, 0, 0, 1], [.25, .25, .5, 0]
        else:
            names = ("Nitrogen", "Methane", "CarbonDioxide", "n-Decane")
            injection, initial = [1, 0, 0, 0], [0, .25, .25, .5]
        return _table_case(names, injection, initial, 5700 * PSIA, 352.594)

    if key == "bakken_light_5comps":
        tc = np.array([343.08, 653.94, 920.808, 1042.812, 1419.732]) * RANKINE
        pc = np.array([667.38, 625.338, 496.272, 454.377, 317.226]) * PSIA
        vc = np.array([1.586, 3.156, 5.347, 6.507, 14.749]) * .06242795996802 / 1000
        mw = np.array([16.04, 42.82, 83.74, 105.91, 200.0]) / 1000
        bic = np.array([
            [0, .0078, .0242, .0324, .0779],
            [.0078, 0, .0046, .0087, .0384],
            [.0242, .0046, 0, .0006, .0169],
            [.0324, .0087, .0006, 0, .0111],
            [.0779, .0384, .0169, .0111, 0],
        ])
        fluid = _mixture(
            ["C1", "C2-C4", "C5-C7", "C8-C9", "C10+"],
            tc, pc, vc, [.008, .1432, .2474, .2861, .6869], mw, bic,
            "Bakken light shale",
        )
        return fluid, _info(
            [1, 0, 0, 0, 0], [.2506, .22, .2, .13, .1994],
            4000 * PSIA, 352.594, parachor=[77, 145.2, 250, 306, 686.3],
        )

    if key in {"barnett3comps_modified_2", "barnett3comps_nat_vars"}:
        names = ("Methane", "Ethane", "n-Propane", "CarbonDioxide")
        fluid = SorbedCompositionalFluid(
            names,
            [[10769611, 5591648, 5819175, 10769611], [2.99, 4.86, 9.56, 2.99]],
        )
        fluid.set_binary_interaction([
            [0, .005, .01, 0], [.005, 0, .01, 0],
            [.01, .01, 0, 0], [0, 0, 0, 0],
        ])
        return fluid, _info(
            [1, 0, 0, 0], [.991, .0088, .0002, 0], 5000 * PSIA, 355.84,
        )

    if key == "barnett3comps_modified":
        names = ("CarbonDioxide", "Methane", "Ethane", "n-Propane")
        fluid = SorbedCompositionalFluid(
            names,
            [[10769611, 10769611, 5591648, 5819175], [2.99, 2.99, 4.86, 9.56]],
        )
        fluid.set_binary_interaction([
            [0, 0, 0, 0], [0, 0, .005, .01],
            [0, .005, 0, .005], [0, .01, .005, 0],
        ])
        return fluid, _info(
            [0, 0, 0, 1], [0, .991, .0088, .0002], 5000 * PSIA, 355.84,
        )

    try:
        return get_main_shale_comp_fluid_case(key)
    except ValueError as error:
        raise ValueError(f"unknown shale compositional fluid case {name!r} in extended registry") from error


__all__ = ["get_shale_comp_fluid_case"]
