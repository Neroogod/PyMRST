"""PyMRST: Python port of the MATLAB Reservoir Simulation Toolbox (MRST 2024a).

Subpackages mirror the MRST source layout so the two codebases and the MRST
textbooks (Lie 2019; Lie & Moyner 2021) can be read side by side:

- gridprocessing: grid constructors and the unstructured grid structure
- params: rock properties, wells, boundary conditions, gravity
- solvers: transmissibilities and pressure/transport solvers
- plotting: visualization of cell and face data
- utils: units, automatic differentiation, array helpers
- modules: ports of MRST add-on modules (ad_core, ad_blackoil, ...)

Convention: all quantities are strict SI and all indices are 0-based.
Boundary faces are marked with -1 in faces.neighbors (MRST uses 0).
"""

from pymrst.utils import units

__version__ = "0.1.0"

__all__ = ["units", "__version__"]
