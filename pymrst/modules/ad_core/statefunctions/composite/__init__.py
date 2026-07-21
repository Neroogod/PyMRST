"""MRST ad-core composite coupling terms (`.../statefunctions/composite/`).

The state functions that link a `CompositeModel`'s submodels: `CouplingTerm`
(the base source-insertion mechanism) and `SourceCouplingTerm` (a conservative
transfer between two submodels sharing component names). `StateFunctionDependent`
is the marker mixin MRST tags derived state functions with.
"""

from pymrst.modules.ad_core.statefunctions.composite.coupling_term import (
    CouplingTerm,
    SourceCouplingTerm,
    StateFunctionDependent,
)

__all__ = ["CouplingTerm", "SourceCouplingTerm", "StateFunctionDependent"]
