"""Helpers private to the simple deck reader (MRST's ``private/`` directory).

MATLAB's `private/` makes these visible only to the parent directory. Python
has no equivalent, so the directory is mirrored for layout and the modules are
ordinary; treat them as internal to `deckinput_simple` regardless.
"""

from pymrst.utils.inout.eclipse.deckinput_simple.private.read_equil import read_equil
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_operator import (
    read_operator,
)
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_operator_mod_field import (
    read_operator_mod_field,
)
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_record_string import (
    read_record_string,
)
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_faults import read_faults
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_multfelt import (
    read_multfelt,
)

__all__ = ["read_equil", "read_faults", "read_multfelt",
           "read_operator", "read_operator_mod_field",
           "read_record_string"]
