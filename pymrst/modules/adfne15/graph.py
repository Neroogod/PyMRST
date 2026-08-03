"""Port of ADFNE15 ``Graph.m``."""
from dataclasses import dataclass
import numpy as np
from .orientation import orientation
from .snap import snap
@dataclass(frozen=True, slots=True)
class GraphNode:
    edges: tuple; neighbours: tuple; location: np.ndarray; type: int; pressure: float=np.nan
@dataclass(frozen=True, slots=True)
class GraphEdge:
    nodes: tuple; type: int; depth: float; length: float; aperture: float; dip: float; direction: float
    pressure_drop: float=np.nan; flow: float=np.nan; hydraulic_gradient: float=np.nan; permeability: float=np.nan; force: float=np.nan
@dataclass(frozen=True, slots=True)
class GraphModel:
    nodes: tuple; edges: tuple; boundary_elements: np.ndarray; boundary_values: np.ndarray|None=None
def graph(model, aperture=lambda length:1e-4, depth=None, tolerance=1e-12):
    lines=np.asarray(model.lines,float)
    if not len(lines): return GraphModel((),(),model.pipe_model.boundary_elements)
    endpoints=lines.reshape(-1,3); q=np.round(endpoints/tolerance).astype(np.int64)
    _,first,inv=np.unique(q,axis=0,return_index=True,return_inverse=True); order=np.argsort(first); keys=endpoints[first[order]]
    remap=np.empty(len(first),int); remap[order]=np.arange(len(order)); pairs=remap[inv].reshape(-1,2)
    types=model.pipe_model.types[model.mask]; depths=model.pipe_model.depths[model.mask]; incident=[[] for _ in keys]
    for e,pair in enumerate(pairs): incident[pair[0]].append(e); incident[pair[1]].append(e)
    edges=[]
    for i,(line,pair) in enumerate(zip(lines,pairs)):
        length=float(np.linalg.norm(line[3:]-line[:3])); orient=orientation(line.reshape(1,6))
        edges.append(GraphEdge(tuple(map(int,pair)),int(types[i]),float(depth(length) if callable(depth) else depths[i]),length,float(aperture(length)),float(orient.dip[0]),float(orient.direction[0])))
    nodes=[]; be=model.pipe_model.boundary_elements
    for i,loc in enumerate(keys):
        neighbours=tuple(int(pairs[e,1] if pairs[e,0]==i else pairs[e,0]) for e in incident[i]); et=[types[e] for e in incident[i] if types[e]!=0]; typ=int(et[0]) if et else 0
        if typ==0 and isinstance(be,np.ndarray) and len(be):
            close=np.flatnonzero(snap(loc.reshape(1,3),be,0,distances_only=True)[0]<=tolerance)
            if len(close): typ=int(close[0]+1)
        nodes.append(GraphNode(tuple(incident[i]),neighbours,loc.copy(),typ))
    return GraphModel(tuple(nodes),tuple(edges),be)
