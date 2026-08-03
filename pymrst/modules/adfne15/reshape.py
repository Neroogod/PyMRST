"""Port of ADFNE15 ``Reshape.m`` row-wise behavior."""
import numpy as np
def reshape(values,rows,columns): return np.asarray(values).reshape(rows,columns,order="C")
