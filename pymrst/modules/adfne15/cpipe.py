"""Native fluent-interface port of ADFNE15 ``CPipe.m``."""

from dataclasses import dataclass
from .pipe import pipe


@dataclass(frozen=True, slots=True)
class CPipe:
    dfn: object

    @classmethod
    def create(cls, *args, **kwargs): return cls(pipe(*args, **kwargs))

    def Backbone(self, *args, **kwargs):
        from .cbackbone import CBackbone
        return CBackbone.create(self.dfn, *args, **kwargs)
