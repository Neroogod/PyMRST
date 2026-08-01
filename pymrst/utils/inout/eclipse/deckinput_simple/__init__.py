"""The simple deck reader (MRST ``utils/inout/eclipse/deckinput_simple``).

MRST keeps two ECLIPSE readers: this one, for GRDECL grid files and the
keywords around them, and the fuller `modules/deckformat` for complete decks.
They are not versions of each other - `readGRDECL` here is referenced from 72
files across MRST - and MRST duplicates some helpers (`readVector.m`) between
the two rather than sharing them.

PyMRST had these three filed under `modules/deckformat` and `gridprocessing`.
They now sit where MRST puts them, with re-exporting shims left at the old
paths so no caller had to change.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
    apply_operator_simple,
)
from pymrst.utils.inout.eclipse.deckinput_simple.cut_grdecl import cut_grdecl
from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
    DeckCursor,
    read_defaulted_record,
)
from pymrst.utils.inout.eclipse.deckinput_simple.read_grdecl import read_grdecl
from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
    read_keywords,
)

__all__ = ["apply_operator_simple", "cut_grdecl", "read_defaulted_record", "read_grdecl", "read_keywords", "DeckCursor"]
