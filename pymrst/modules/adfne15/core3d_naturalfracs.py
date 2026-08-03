"""Callable port of ADFNE15 ``Core3D_NaturalFracs.m``."""
from dataclasses import dataclass
from .dfn_bin import dfn_bin
from .getshapepoly import get_shape_poly
from .intersect import intersect
@dataclass(frozen=True,slots=True)
class Core3DResult: boundaries:tuple; fractures:tuple; intersections:object
def core3d_natural_fractures(*,count=99,seed=1234567890,resolution=20):
    boundaries=tuple(get_shape_poly(radius=.5,center=[.5,1,1],length=2,resolution=resolution))
    result=dfn_bin(dim=3,n=count,minl=.1,maxl=.3,mu=.2,bbx=[.45,.45,.45,2.55,1.55,1.55],polyDomain=boundaries,shape='c',q=12,seed=seed)
    return Core3DResult(boundaries,result.fractures,intersect(result.fractures))
