"""Tests for MRST ``getInitializationRegionsDeck`` region splitting."""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.initialization import getInitializationRegionsDeck


def test_eql_sat_pvt_combinations_are_split_and_mapped():
    cells = SimpleNamespace(num=6, indexMap=np.arange(6))
    model = SimpleNamespace(
        G=SimpleNamespace(cells=cells),
        fluid=SimpleNamespace(rho_s=np.array([1000.0, 800.0])),
        n_phases=2,
        water=True,
        oil=True,
        gas=False,
        raw_fluid=SimpleNamespace(swcon=0.0),
    )
    deck = {
        "SOLUTION": {
            "EQUIL": np.array([
                [1000.0, 1.0e7, 1100.0, 5.0, 900.0, 0.0],
                [1200.0, 1.2e7, 1300.0, 7.0, 1000.0, 0.0],
            ])
        },
        "REGIONS": {
            "EQLNUM": [1, 1, 1, 2, 2, 2],
            "SATNUM": [1, 1, 2, 1, 1, 1],
            "PVTNUM": [1, 2, 1, 1, 1, 2],
        },
    }
    regions = getInitializationRegionsDeck(model, deck)
    assert len(regions) == 5
    assert [r.eqlnum for r in regions] == [1, 1, 1, 2, 2]
    assert {(r.satnum, r.pvtnum) for r in regions[:3]} == {
        (1, 1), (1, 2), (2, 1)
    }
    assert sorted(np.concatenate([r.cells for r in regions]).tolist()) == list(range(6))
    assert np.allclose(regions[0].contacts, [1100.0])
    assert np.allclose(regions[-1].contacts_pc, [7.0])
