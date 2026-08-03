"""Port of ADFNE15 ``Center.m``."""
import numpy as np
from .plane import plane
from .project import project
def _centroid(poly):
    definition=plane(poly) if poly.shape[1]==3 else None; flat=project(poly,definition) if definition is not None else poly; following=np.roll(flat,-1,axis=0); cross=flat[:,0]*following[:,1]-following[:,0]*flat[:,1]; den=3*np.sum(cross)
    if abs(den)<=1e-15: return np.mean(poly,axis=0)
    c=np.sum((flat+following)*cross[:,None],axis=0)/den
    return c if definition is None else definition[:3]+definition[3:6]*c[0]+definition[6:9]*c[1]
def center(items):
    if isinstance(items,(list,tuple)): return np.vstack([_centroid(np.asarray(v,float)) for v in items])
    v=np.asarray(items,float); columns=v.shape[1]
    if columns>2 and columns%2==0: half=columns//2; return .5*(v[:,:half]+v[:,half:])
    return np.mean(v,axis=0)
