"""GRDECL readers for the MRST `deckformat` subset."""

from pymrst.modules.deckformat.grdecl.read_grdecl import read_grdecl
from pymrst.modules.deckformat.grdecl.read_keywords import read_keywords

__all__ = ["read_grdecl", "read_keywords"]

