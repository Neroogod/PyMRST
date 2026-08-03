"""Port of the 2-D center path in ADFNE15 ``Pipe.m``."""
from dataclasses import dataclass
import numpy as np
from .split import split
@dataclass(frozen=True, slots=True)
class PipeModel:
    fractures: object; intersections: np.ndarray; intersection_pairs: np.ndarray
    labels: np.ndarray; pipes: np.ndarray; types: np.ndarray; depths: np.ndarray
    boundary_elements: object; source_ids: np.ndarray; percolating: bool
def pipe(boundary_elements, fractures, method="center"):
    if str(method).lower().startswith("tri"):
        from .mesh import mesh
        return mesh([], 0.1, False, str(method)[3:] or "r")
    if not str(method).lower().startswith(("cnt", "center")): raise ValueError(f"unknown Pipe method: {method}")
    if isinstance(boundary_elements, (list, tuple)):
        from .center import center
        from .intersect import intersect
        boundaries=tuple(np.asarray(value,float) for value in boundary_elements)
        network=boundaries+tuple(np.asarray(value,float) for value in fractures)
        result=intersect(network); centers=center(network)
        pipes=[]; types=[]; depths=[]; sources=[]
        for geometry,(first,second) in zip(result.geometry,result.pairs):
            midpoint=.5*(geometry[:3]+geometry[3:]); depth=float(np.linalg.norm(geometry[3:]-geometry[:3]))
            for owner in (first,second):
                pipes.append(np.r_[centers[owner],midpoint]); types.append(owner+1 if owner<len(boundaries) else 0); depths.append(max(depth,1e-14)); sources.append(owner)
        percolating=len(boundaries)>=2 and result.labels[0]>0 and result.labels[0]==result.labels[1]
        return PipeModel(network,result.geometry,result.pairs,result.labels,
                         np.asarray(pipes,float).reshape(-1,6),np.asarray(types,int),
                         np.asarray(depths,float),boundaries,np.asarray(sources,int),bool(percolating))
    be=np.atleast_2d(np.asarray(boundary_elements,float)); network=np.vstack((be,np.atleast_2d(np.asarray(fractures,float))))
    if network.shape[1]!=4: raise ValueError("current Pipe owner requires 2-D lines")
    r=split(network); p=np.column_stack((r.lines[:,:2],np.zeros(len(r.lines)),r.lines[:,2:],np.zeros(len(r.lines))))
    typ=np.zeros(len(p),int)
    for i in range(len(be)): typ[r.source_ids==i]=i+1
    be3=np.column_stack((be[:,:2],np.zeros(len(be)),be[:,2:],np.zeros(len(be))))
    perc=len(be)>=2 and r.labels[0]>0 and r.labels[0]==r.labels[1]
    return PipeModel(network,r.intersections,r.intersection_pairs,r.labels,p,typ,np.ones(len(p)),be3,r.source_ids,bool(perc))
