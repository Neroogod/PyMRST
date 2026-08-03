"""Port of ADFNE15 ``Filter.m``."""
import numpy as np
def filter_ids(indices,cardinality=2):
    values,counts=np.unique(np.asarray(indices).reshape(-1),return_counts=True)
    return values[counts>=cardinality]
