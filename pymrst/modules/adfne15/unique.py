"""Port of ADFNE15 ``Unique.m``."""
import numpy as np
def unique_rows(values,tolerance=1e-14,*,return_indices=False):
    values=np.asarray(values,float); q=np.round(values/tolerance).astype(np.int64)
    _,first,inverse=np.unique(q,axis=0,return_index=True,return_inverse=True); order=np.sort(first); out=values[order]
    remap=np.empty(len(first),int)
    for new,old in enumerate(order): remap[np.flatnonzero(first==old)[0]]=new
    inverse=remap[inverse]
    return (out,order,inverse) if return_indices else out
