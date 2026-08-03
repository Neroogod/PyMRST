"""Port of ADFNE15 ``Solve.m`` cubic-law graph flow."""
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .boundary import boundary
from .graph import GraphModel
def conductance(length, aperture, depth, kinematic_viscosity=1.787e-6):
    return 9.8*np.asarray(depth)*np.asarray(aperture)**3/(12*kinematic_viscosity*np.asarray(length))
def solve(model, boundary_values=None, kinematic_viscosity=1.787e-6):
    if not model.nodes: return model
    maximum=max(node.type for node in model.nodes); values=np.arange(maximum,dtype=float) if boundary_values is None else np.asarray(boundary_values,float)
    if len(values)<maximum: raise ValueError("boundary value count is too small")
    pressure=np.full(len(model.nodes),np.nan); valid=np.isfinite(values)
    for i,node in enumerate(model.nodes):
        if node.type:
            value=values[node.type-1]; pressure[i]=value if np.isfinite(value) else boundary(node.location.reshape(1,3),model.boundary_elements[valid],values[valid])[0]
    inner=np.flatnonzero([node.type==0 for node in model.nodes]); lookup={n:i for i,n in enumerate(inner)}; rows=[]; cols=[]; data=[]; rhs=np.zeros(len(inner))
    for row,nid in enumerate(inner):
        total=0.0
        for eid in model.nodes[nid].edges:
            edge=model.edges[eid]; other=edge.nodes[1] if edge.nodes[0]==nid else edge.nodes[0]; c=float(conductance(edge.length,edge.aperture,edge.depth,kinematic_viscosity)); total+=c
            if other in lookup: rows.append(row); cols.append(lookup[other]); data.append(-c)
            else: rhs[row]+=c*pressure[other]
        rows.append(row); cols.append(row); data.append(total)
    if len(inner): pressure[inner]=spla.spsolve(sp.csr_matrix((data,(rows,cols)),shape=(len(inner),len(inner))),rhs)
    nodes=tuple(replace(node,pressure=float(pressure[i])) for i,node in enumerate(model.nodes)); edges=[]
    for edge in model.edges:
        first,second=edge.nodes; drop=pressure[second]-pressure[first]
        if drop>0: first,second,drop=second,first,-drop
        gradient=drop/edge.length; flow=float(conductance(edge.length,edge.aperture,edge.depth,kinematic_viscosity)*drop); area=edge.aperture*edge.depth
        permeability=0.0 if flow==0 else abs(flow*kinematic_viscosity/(gradient*area))
        edges.append(replace(edge,nodes=(first,second),pressure_drop=float(drop),flow=flow,hydraulic_gradient=float(gradient),permeability=float(permeability),force=float(-area*drop)))
    return GraphModel(nodes,tuple(edges),model.boundary_elements,values.copy())
