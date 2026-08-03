"""Callable port of ADFNE15 ``NaturalFrac2D_Direction.m``."""
import numpy as np
from .dfn_bin import dfn_bin
from .length import lengths
def natural_fractures_2d_direction(*,count=150,seed=None):
    result=dfn_bin(dim=2,n=count,bbx=[10,10,190,130],minl=5,mu=10,maxl=25,dir=45,ddir=-15,seed=seed)
    return result,float(np.sum(lengths(result.fractures)))
