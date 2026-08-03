"""Port of ADFNE15 ``Not.m`` with Python zero-based indices."""
import numpy as np
def not_indices(values,indices):
    values=np.asarray(values); mask=np.ones(values.size,bool); mask[np.asarray(indices,int)]=False
    return values.reshape(-1)[mask]
