"""Examples that demonstrate multiple aspects of the PyMRST grid structure.

Port of MRST's ``examples/grids/``. Each tutorial builds grids a different way
and inspects the resulting structure, so the natural check is the structure
itself: cell and face counts, `facePos` monotonicity, volumes and centroids,
and the mapping a subgrid keeps back to its parent. Those compare exactly,
which makes these tutorials better regression tests than their figures suggest.
"""

__all__: list[str] = []
