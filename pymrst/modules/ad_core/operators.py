"""Backward-compatible import path for the AD-core TPFA operators.

The implementation lives in `ad_core.utils.operators`, mirroring MRST's
`ad-core/utils/setupOperatorsTPFA.m`.
"""

from pymrst.modules.ad_core.utils.operators import Operators, setup_operators_tpfa

__all__ = ["Operators", "setup_operators_tpfa"]
