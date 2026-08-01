"""Moved to `pymrst.utils.inout.eclipse.deckinput_simple.read_grdecl`.

MRST keeps `readGRDECL` in `utils/inout/eclipse/deckinput_simple/`, not in
`modules/deckformat`, so that is where the implementation now lives.

This module re-exports it so existing imports keep working; new code
should import from `pymrst.utils.inout.eclipse.deckinput_simple`.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.read_grdecl import (  # noqa: F401
    read_grdecl,
)

__all__ = ['read_grdecl']
