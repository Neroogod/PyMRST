"""Compatibility landing file for MRST `TransportOverallCompositionModel.m`.

PyMRST exposes the first transport slice as
`OverallCompositionModel.transport_step`.
"""

from pymrst.modules.compositional.model import (
    CompositionalState,
    OverallCompositionModel,
)


class TransportOverallCompositionMixin:
    """Marker for the MRST sequential transport layout.

    The implementation currently lives on `OverallCompositionModel`; this class
    exists only so imports through the MRST-shaped layout have a stable symbol.
    """


__all__ = [
    "OverallCompositionModel", "CompositionalState",
    "TransportOverallCompositionMixin",
]
