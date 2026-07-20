"""MRST `ad-blackoil/components/OilComponent.m`."""

from typing import Any, List, Optional

from pymrst.modules.ad_core.statefunctions.components.components import Component


class OilComponent(Component):
    """Oil: in the oil phase, and vaporised into the gas through `rv`."""

    def __init__(self, name: str = "oil", oil_index: int = 1,
                 gas_index: int = 2):
        super().__init__(name)
        self.oil_index = oil_index
        self.gas_index = gas_index

    def component_density(self, model, state, b, rs=None,
                          rv=None) -> List[Optional[Any]]:
        out: List[Optional[Any]] = [None] * len(b)
        out[self.oil_index] = b[self.oil_index]
        if rv is not None:
            out[self.gas_index] = rv * b[self.gas_index]
        return out


__all__ = ["OilComponent"]
