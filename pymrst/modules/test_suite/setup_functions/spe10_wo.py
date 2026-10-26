"""MRST ``test-suite/setup-functions/spe10_wo.m``.

The test case 'spe10_wo': SPE10 Model 2 (Christie & Blunt 2001) as a water-oil
model from `setupSPE10_AD`, with a subset of its layers (`layers`: numbers in
1..85, or 'tarbert' for 1..35 and 'upper_ness' for 36..85; all by default),
gravity on. Other options are passed to `setupSPE10_AD`. Plot options give a
perspective view in 3D and a top view in 2D.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.spe10.setup_spe10_ad import setup_spe10_ad
from pymrst.modules.test_suite.utils.pack_test_case_setup import pack_test_case_setup
from pymrst.modules.test_suite.utils.process_test_case_input import process_test_case_input
from pymrst.params.gravity.gravity import gravity

DESCRIPTION = ("SPE10 Model 2 (Christie & Blunt, 2008, SPE Res. Eval. & Eng., "
               "doi: 10.2118/72469-pa) with support for subset of layers")
ALIASES = {"tarbert": np.arange(1, 36), "upper_ness": np.arange(36, 86)}


def spe10_wo(*args: Any, **varargin: Any):
    """MRST `setup = spe10_wo(fullSetup, ...)`."""
    options, full_setup, setup = process_test_case_input("spe10_wo", {"layers": []},
                                                         DESCRIPTION, *args, **varargin)
    layers = options["layers"]
    if layers is None or np.size(layers) == 0:
        options["layers"] = np.arange(1, 86)
    elif isinstance(layers, str):
        if layers not in ALIASES:
            raise ValueError("Optional input argument `layers` must be either a vetor of "
                             "layer numbers, or one of the aliases `tarbert` (= layer 1:35) "
                             "or `upper_ness` (= layer 35:85)")
        options["layers"] = ALIASES[layers]
    else:
        options["layers"] = np.atleast_1d(layers)
        assert options["layers"].min() >= 1 and options["layers"].max() <= 85
    if not full_setup:
        setup.options = options
        return setup
    gravity("reset", "on")
    state0, model, schedule = setup_spe10_ad(**options["extra"], layers=options["layers"])
    if model.G.griddim == 3 and model.G.cartDims[2] > 1:
        view, proj, size = [120, 26], "perspective", [800, 400]
    else:
        view, proj, size = [0, 90], "orthographic", [500, 800]
    zpba = 0.3*np.size(options["layers"])/85
    plot_options = ["PlotBoxAspectRatio", [1, 1.83, zpba], "Projection", proj,
                    "View", view, "Size", size]
    return pack_test_case_setup("spe10_wo", description=DESCRIPTION, options=options,
                                state0=state0, model=model, schedule=schedule,
                                plotOptions=plot_options)


__all__ = ["spe10_wo"]


if __name__ == "__main__":
    import warnings

    from pymrst.params.gravity.gravity import isolated_gravity

    def test_light_setup_and_aliases():
        setup = spe10_wo(False, layers="tarbert")
        assert setup.name == "spe10_wo" and list(setup.options["layers"]) == list(range(1, 36))
        assert setup.model is None
        print("ok test_light_setup_and_aliases")

    def test_one_2d_layer():
        with isolated_gravity(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup = spe10_wo(layers=85, J=np.arange(1, 11), make2D=True)
            assert gravity()[2] > 0
        assert setup.model.G.griddim == 2 and tuple(setup.model.G.cartDims) == (60, 10)
        assert setup.plotOptions[setup.plotOptions.index("View") + 1] == [0, 90]
        print("ok test_one_2d_layer")

    test_light_setup_and_aliases()
    test_one_2d_layer()
    print("\nAll tests passed!")
