"""Port of ADFNE15 ``Polyline.m``."""
import numpy as np
def polyline(polygon):
    p=np.asarray(polygon)
    if p.ndim==0 or p.size==1: n=int(p.reshape(-1)[0]); return np.column_stack((np.arange(n-1),np.arange(1,n)))
    return np.hstack((p,np.roll(p,-1,axis=0)))
