"""MRST ``ad-core/statefunctions/flowprops/BlackOilShrinkageFactors.m``.

Shrinkage factors for a black-oil fluid: live oil depends on `rs` and wet gas
on `rv`, each with a flag saying whether the cell is saturated (the other
hydrocarbon phase is present).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.shrinkage_factors import (
    ShrinkageFactors,
)
from pymrst.modules.ad_core.statefunctions.state_function import STATE
from pymrst.utils.adi import value


class BlackOilShrinkageFactors(ShrinkageFactors):
    """`bO(p, rs, sG > 0)` and `bG(p, rv, sO > 0)`; others by pressure."""

    useSaturatedFlag = True

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, regions)
        from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
            ThreePhaseBlackOilModel,
        )
        assert isinstance(model, ThreePhaseBlackOilModel), (
            "Model must be derived from the black-oil model. Did you want the "
            "regular ShrinkageFactors class instead?")
        self.disgas = bool(model.disgas)
        self.vapoil = bool(model.vapoil)
        if self.disgas:
            self.depends_on(["rs"], STATE)
        if self.vapoil:
            self.depends_on(["rv"], STATE)
        if self.useSaturatedFlag:
            self.depends_on(["s"], STATE)

    def evaluate_phase_shrinkage_factor(self, model: Any, state: Any,
                                        name: str, p: Any) -> Any:
        if self.disgas and name == "O":
            rs = model.get_prop(state, "rs")
            flag = (np.asarray(value(model.get_prop(state, "sg"))) > 0
                    if self.useSaturatedFlag
                    else np.zeros(np.size(value(rs)), dtype=bool))
            return self.evaluate_fluid(model, "bO", p, rs, flag)
        if self.vapoil and name == "G":
            rv = model.get_prop(state, "rv")
            flag = (np.asarray(value(model.get_prop(state, "so"))) > 0
                    if self.useSaturatedFlag
                    else np.zeros(np.size(value(rv)), dtype=bool))
            return self.evaluate_fluid(model, "bG", p, rv, flag)
        return super().evaluate_phase_shrinkage_factor(model, state, name, p)

    evaluatePhaseShrinkageFactor = evaluate_phase_shrinkage_factor


__all__ = ["BlackOilShrinkageFactors"]


if __name__ == "__main__":
    def test_a_model_that_is_not_black_oil_is_refused():
        from pymrst.modules.ad_core.ad_oo.testing import toy_model
        try:
            BlackOilShrinkageFactors(toy_model("WOG", nx=2))
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_a_model_that_is_not_black_oil_is_refused")

    test_a_model_that_is_not_black_oil_is_refused()
    print("\nAll tests passed!")
