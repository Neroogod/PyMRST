"""Moved to `pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record`.

MRST keeps `readDefaultedRecord` in `utils/inout/eclipse/deckinput_simple/`,
not in `modules/deckformat`, so that is where the implementation now lives.

This module re-exports it so existing imports keep working; new code
should import from `pymrst.utils.inout.eclipse.deckinput_simple`.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (  # noqa: F401
    read_defaulted_record,
    DeckCursor,
    expand_defaults,
    read_defaulted_kw,
    read_fixed_num_records,
    strip_quotes,
    tokenize,
    tokenize_with_lines,
)

__all__ = ['read_defaulted_record', 'DeckCursor', 'expand_defaults', 'read_defaulted_kw', 'read_fixed_num_records', 'strip_quotes', 'tokenize', 'tokenize_with_lines']
