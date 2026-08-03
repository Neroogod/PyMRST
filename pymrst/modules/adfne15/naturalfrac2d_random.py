"""Callable port of ADFNE15 ``NaturalFrac2D_Random.m``."""
import numpy as np
from .dfn_orig import dfn_orig
from .length import lengths
def natural_fractures_2d_random(*,count=150,seed=None):
    result=dfn_orig(dim=2,n=count,bbx=[10,10,190,130],minl=5,mu=10,maxl=25,dir=0,ddir=-1e-9,seed=seed)
    return result,float(np.sum(lengths(result.fractures)))
