"""Moved to `pymrst.utils.inout.eclipse.deckinput_simple.cut_grdecl`.

MRST keeps `cutGrdecl` in `utils/inout/eclipse/deckinput_simple/`, not in
`gridprocessing`, so that is where the implementation now lives.

This module re-exports it so existing imports keep working; new code
should import from `pymrst.utils.inout.eclipse.deckinput_simple`.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.cut_grdecl import (  # noqa: F401
    cut_grdecl,
)

__all__ = ['cut_grdecl']
