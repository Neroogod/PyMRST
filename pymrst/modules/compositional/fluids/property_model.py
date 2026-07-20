"""Port of MRST ``fluids/PropertyModel.m``."""

from __future__ import annotations


class PropertyModel:
    """Base class for compositional property models.

    MRST's class only stores the associated compositional fluid.  PyMRST keeps
    the same minimal public surface and uses it as the base for concrete
    property-model implementations.
    """

    def __init__(self, fluid=None, **kwargs):
        if fluid is None:
            fluid = kwargs.pop("f", kwargs.pop("mixture", None))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        self.fluid = fluid

    @property
    def mixture(self):
        """Pythonic alias for MRST's ``fluid`` property."""
        return self.fluid

    @mixture.setter
    def mixture(self, value) -> None:
        self.fluid = value


__all__ = ["PropertyModel"]
