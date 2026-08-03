"""Port of ADFNE15 ``Length.m``."""
import numpy as np
from .center import center
def lengths(items):
    if isinstance(items,(list,tuple)): return np.asarray([2*np.mean(np.linalg.norm(np.asarray(p)-center(np.asarray(p)),axis=1)) for p in items])
    lines=np.atleast_2d(np.asarray(items,float)); d=lines.shape[1]//2; return np.linalg.norm(lines[:,d:]-lines[:,:d],axis=1)
