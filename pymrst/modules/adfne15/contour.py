"""Native Matplotlib port of ADFNE15 ``Contour.m``."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import griddata
from .idw import inverse_distance_weighting
@dataclass(frozen=True,slots=True)
class ContourResult: x:np.ndarray; y:np.ndarray; values:np.ndarray; artist:object=None
def contour(points,values,resolution=20,filled=True,method='idw',*,axis=None,**kwargs):
    points=np.asarray(points,float); values=np.asarray(values,float).reshape(-1)
    x,y=np.meshgrid(np.linspace(points[:,0].min(),points[:,0].max(),resolution),np.linspace(points[:,1].min(),points[:,1].max(),resolution))
    targets=np.column_stack((x.ravel(),y.ravel()))
    z=(inverse_distance_weighting(points[:,:2],values,targets) if method.lower()=='idw' else griddata(points[:,:2],values,targets,method=method)).reshape(x.shape)
    artist=None
    if axis is not None: artist=(axis.contourf if filled else axis.contour)(x,y,z,**kwargs)
    return ContourResult(x,y,z,artist)
