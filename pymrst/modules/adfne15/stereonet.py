"""Native computational/Matplotlib port of ADFNE15 ``Stereonet.m``."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True,slots=True)
class StereonetResult: x:np.ndarray; y:np.ndarray; density:np.ndarray|None; artist:object=None
def stereonet(dip,direction,*,density=False,ndip=3,ndir=12,axis=None,**kwargs):
    radius=np.asarray(dip,float)/90; theta=np.deg2rad(direction); x=radius*np.cos(theta); y=radius*np.sin(theta); counts=None
    if density:
        counts,_,_=np.histogram2d(radius,theta%(2*np.pi),bins=(np.linspace(0,1,ndip+1),np.linspace(0,2*np.pi,ndir+1)))
    artist=axis.scatter(x,y,**kwargs) if axis is not None else None
    return StereonetResult(x,y,counts,artist)
