"""Native fluent-interface port of ADFNE15 ``CGraph.m``."""

from dataclasses import dataclass
from .graph import graph
from .solve import solve


@dataclass(frozen=True, slots=True)
class CGraph:
    dfn: object

    @classmethod
    def create(cls, *args, **kwargs): return cls(graph(*args, **kwargs))

    def Solve(self, *args, **kwargs): return solve(self.dfn, *args, **kwargs)
