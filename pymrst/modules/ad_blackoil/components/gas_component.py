"""MRST `ad-blackoil/components/GasComponent.m`."""

from typing import Any, List, Optional

from pymrst.modules.ad_core.components import Component


class GasComponent(Component):
    """Gas: in the gas phase, and dissolved in the oil through `rs`."""

    def __init__(self, name: str = "gas", gas_index: int = 2,
                 oil_index: int = 1):
        super().__init__(name)
        self.gas_index = gas_index
        self.oil_index = oil_index

    def component_density(self, model, state, b, rs=None,
                          rv=None) -> List[Optional[Any]]:
        out: List[Optional[Any]] = [None] * len(b)
        out[self.gas_index] = b[self.gas_index]
        if rs is not None:
            out[self.oil_index] = rs * b[self.oil_index]
        return out


__all__ = ["GasComponent"]
