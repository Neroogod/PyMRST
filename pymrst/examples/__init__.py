"""Worked examples demonstrating features of PyMRST.

Port of MRST's top-level ``examples/`` tree, which holds the tutorials that
introduce the toolbox rather than any one module's own examples. Module-level
examples stay with their modules, as in MRST.

The layout mirrors MRST:

===========================  ====================================
MRST                         PyMRST
===========================  ====================================
``examples/``                ``pymrst/examples/``
``examples/grids/``          ``pymrst/examples/grids/``
``examples/datasets/``       ``pymrst/examples/datasets/``
===========================  ====================================

Every module here runs standalone and ends with its own unit tests, so the
tutorials double as regression coverage rather than only as documentation.
Where MRST's version ends in a plotting call, the port computes the same
quantities and checks them; the figures are optional and guarded so the
examples run headless.

See ``examples_parity.md`` for status and ``examples_implementation.md`` for
the porting plan.
"""

__all__: list[str] = []
