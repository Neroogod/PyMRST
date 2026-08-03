"""Native fluent-interface port of ADFNE15 ``CBackbone.m``."""

from dataclasses import dataclass
from .backbone import backbone


@dataclass(frozen=True, slots=True)
class CBackbone:
    dfn: object

    @classmethod
    def create(cls, *args, **kwargs): return cls(backbone(*args, **kwargs))

    def Graph(self, *args, **kwargs):
        from .cgraph import CGraph
        return CGraph.create(self.dfn, *args, **kwargs)
