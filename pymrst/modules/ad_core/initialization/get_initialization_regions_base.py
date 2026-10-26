"""MRST ``ad-core/initialization/getInitializationRegionsBase.m``.

An equilibration region: datum pressure (1 atm) and depth (0), the phase
whose pressure the datum gives (`reference_index`, by default the phase
after water), the cells (all), a density function `rho{i}(p, z)` per phase,
one contact depth per non-reference phase with the capillary pressure
there (`contacts_pc`, 0), the saturation and PVT region (1), and per phase
a capillary curve `pc_functions`, its sign and scaling, and saturation
bounds `s_min`/`s_max` (0 and 1). Options override any of these; an unknown
option is an error, as with MATLAB's `merge_options`.

`get_initialization_regions_base_ad` (in `regions.py`) is PyMRST's earlier
version.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from pymrst.utils.units import ATM


def get_initialization_regions_base(model: Any, rho: Sequence[Any], contacts: Sequence[float],
                                    /, **options: Any) -> SimpleNamespace:
    """MRST `region = getInitializationRegionsBase(model, rho, contacts, ...)`."""
    nph = int(np.sum(model.get_active_phases()))
    contacts = np.atleast_1d(np.asarray(contacts, dtype=float))
    assert contacts.size == nph - 1
    region = dict(datum_pressure=1*ATM, datum_depth=0.0,
                  reference_index=int(model.water),          # 0-based: MRST's water+1
                  cells=np.arange(model.G.cells.num), rho=list(rho), contacts=contacts,
                  saturation_region=1, pvt_region=1, pc_sign=np.ones(nph),
                  pc_functions=[], pc_scale=np.ones(nph), contacts_pc=np.zeros(nph - 1),
                  s_max=np.ones(nph), s_min=np.zeros(nph))
    for key, item in options.items():
        if key not in region:
            raise KeyError(f"getInitializationRegionsBase has no option {key!r}")
        region[key] = item
    region = SimpleNamespace(**region)
    assert np.isscalar(region.datum_depth) or np.size(region.datum_depth) == 1
    assert np.isscalar(region.datum_pressure) or np.size(region.datum_pressure) == 1
    assert np.size(region.contacts_pc) == contacts.size
    return region


getInitializationRegionsBase = get_initialization_regions_base

__all__ = ["getInitializationRegionsBase", "get_initialization_regions_base"]


if __name__ == "__main__":
    def _model(water=True, oil=True, gas=False, nc=4):
        return SimpleNamespace(water=water, oil=oil, gas=gas,
                               G=SimpleNamespace(cells=SimpleNamespace(num=nc)),
                               get_active_phases=lambda: np.array([water, oil, gas]))

    def test_the_defaults_are_mrsts():
        region = get_initialization_regions_base(_model(), [None, None], [100.0])
        assert region.datum_pressure == ATM and region.reference_index == 1
        assert region.cells.tolist() == [0, 1, 2, 3]
        np.testing.assert_array_equal(region.s_max, [1, 1])
        print("ok test_the_defaults_are_mrsts")

    def test_one_contact_per_non_reference_phase():
        try:
            get_initialization_regions_base(_model(), [None, None], [1.0, 2.0])
        except AssertionError:
            pass
        else:
            raise AssertionError("two contacts for two phases must be refused")
        try:
            get_initialization_regions_base(_model(), [None, None], [1.0], noSuch=1)
        except KeyError:
            pass
        else:
            raise AssertionError("an unknown option must be refused")
        print("ok test_one_contact_per_non_reference_phase")

    test_the_defaults_are_mrsts()
    test_one_contact_per_non_reference_phase()
    print("\nAll tests passed!")
