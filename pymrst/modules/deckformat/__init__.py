"""Minimal ECLIPSE deck input (MRST deckformat subset).

Includes the Norne field model: `fetch_norne` downloads the real corner-point
grid and petrophysics from OPM's `opm-tests` at the same pinned revision MRST
uses, and `read_norne_grdecl` assembles them the way `makeNorneGRDECL.m` does.
"""

from pymrst.modules.deckformat.grdecl import read_grdecl, read_keywords
from pymrst.modules.deckformat.egg import egg_data_directory, fetch_egg_dataset
from pymrst.modules.deckformat.norne import (
    fetch_norne,
    fetch_norne_field_model,
    norne_field_directory,
    norne_output_prefix,
    norne_rock,
    read_norne_field_deck,
    read_norne_grdecl,
)
from pymrst.modules.deckformat.spe1 import read_spe1_deck
from pymrst.modules.deckformat.spe3 import read_spe3_deck
from pymrst.modules.deckformat.spe9 import read_spe9_deck
from pymrst.modules.deckformat.spe10 import (
    fetch_spe10_dataset, load_spe10_model1, spe10_data_directory,
)
from pymrst.modules.deckformat.params.rock import process_aquifer

__all__ = [
    "read_spe1_deck", "read_spe3_deck", "read_spe9_deck",
    "egg_data_directory", "fetch_egg_dataset", "read_grdecl", "read_keywords",
    "fetch_spe10_dataset", "load_spe10_model1", "spe10_data_directory",
    "fetch_norne", "read_norne_grdecl", "norne_rock",
    "fetch_norne_field_model", "norne_field_directory", "norne_output_prefix",
    "read_norne_field_deck",
    "process_aquifer",
]
