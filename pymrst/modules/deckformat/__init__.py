"""Minimal ECLIPSE deck input (MRST deckformat subset).

Includes the Norne field model: `fetch_norne` downloads the real corner-point
grid and petrophysics from OPM's `opm-tests` at the same pinned revision MRST
uses, and `read_norne_grdecl` assembles them the way `makeNorneGRDECL.m` does.
"""

from pymrst.modules.deckformat.grdecl import read_grdecl, read_keywords
from pymrst.modules.deckformat.norne import (
    fetch_norne,
    norne_rock,
    read_norne_grdecl,
)
from pymrst.modules.deckformat.spe1 import read_spe1_deck
from pymrst.modules.deckformat.params.rock import process_aquifer

__all__ = [
    "read_spe1_deck", "read_grdecl", "read_keywords",
    "fetch_norne", "read_norne_grdecl", "norne_rock",
    "process_aquifer",
]
