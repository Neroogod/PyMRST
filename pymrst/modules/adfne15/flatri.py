"""Port of ADFNE15 ``Flatri.m`` using MATLAB column-major extraction."""
import numpy as np
def flatri(values,diagonal=-1):
    values=np.asarray(values); mask=np.tri(*values.shape,k=diagonal,dtype=bool)
    return values.ravel(order="F")[mask.ravel(order="F")]
