"""Moved to `pymrst.utils.inout.eclipse.deckinput_simple.read_keywords`.

`read_grdecl` needs this, and `read_grdecl` now lives under
`utils/inout` where MRST keeps it. Leaving the helper behind would have
`pymrst/utils` importing from `pymrst/modules` - a layering inversion,
and in fact a genuine import cycle, since `modules/deckformat/__init__`
imports the shim that imports back.

MRST has the same duplication: `readVector.m` exists in both
`utils/inout/eclipse/deckinput_simple/private/` and
`modules/deckformat/deckinput/private/`, copied rather than shared.

This module re-exports it so existing imports keep working.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import read_keywords  # noqa: F401

__all__ = ["read_keywords"]
