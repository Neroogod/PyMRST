"""Port of ADFNE15 ``Boundary.m``."""
import numpy as np
from .snap import snap
def boundary(points, elements, values, power=1.0):
    d = snap(points, elements, 0, distances_only=True); v = np.asarray(values, float).reshape(-1)
    # Preserve ADFNE's exact-hit behaviour: 1/0 produces inf and the weighted
    # quotient becomes inf/inf -> NaN. A safer policy belongs in an extension.
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (1.0/d)**power
        return (w@v)/np.sum(w, axis=1)
