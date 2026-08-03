"""End-to-end Pipe/Backbone/Graph/Boundary/Solve qualification."""
import numpy as np
from pymrst.modules.adfne15 import (CBackbone, CGraph, CPipe, backbone, boundary, conductance, graph,
                                    pipe, solve)

BOUNDARIES=np.array([[0,0,0,1],[1,0,1,1]],float)
FRACTURES=np.array([[0,.5,1,.5],[.5,.5,.5,.8]],float)

def test_boundary_inverse_distance_and_exact_hits():
    values=boundary([[0,.25],[.5,.5],[1,.75]],BOUNDARIES,[1,0])
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert values[1] == .5

def test_pipe_backbone_graph_solve_is_conservative_and_directed():
    pipes=pipe(BOUNDARIES,FRACTURES)
    assert pipes.percolating
    skeleton=backbone(pipes)
    assert len(skeleton.lines)<len(pipes.pipes)
    network=graph(skeleton,aperture=lambda length:1e-3)
    solved=solve(network,[1,0])
    pressure=np.array([node.pressure for node in solved.nodes])
    assert np.all(np.isfinite(pressure)) and pressure.min()>=0 and pressure.max()<=1
    for node_index,node in enumerate(solved.nodes):
        if node.type==0:
            signed=0.0
            for edge_index in node.edges:
                edge=solved.edges[edge_index]
                signed += edge.flow if edge.nodes[1]==node_index else -edge.flow
            assert abs(signed)<1e-15
    assert all(edge.pressure_drop<=0 for edge in solved.edges)
    assert all(edge.permeability>=0 for edge in solved.edges)

def test_conductance_uses_adfne_cubic_law():
    expected=9.8*2*(1e-3)**3/(12*1.787e-6*4)
    assert conductance(4,1e-3,2)==expected

def test_nonpercolating_pipe_refuses_to_invent_backbone():
    model=pipe(BOUNDARIES,[[.2,.2,.3,.3]])
    assert not model.percolating
    assert len(backbone(model).lines)==0

def test_3d_polygon_center_pipe_preserves_intersection_depth_and_boundary_types():
    left=np.array([[0,0,0],[0,.4,0],[0,.4,1],[0,0,1]],float)
    right=left.copy(); right[:,0]=1
    fracture=np.array([[0,.1,.5],[1,.1,.5],[1,.8,.5],[0,.8,.5]],float)
    model=pipe([left,right],[fracture],method="center")
    assert model.percolating
    np.testing.assert_array_equal(model.intersection_pairs,[[0,2],[1,2]])
    assert model.pipes.shape==(4,6)
    np.testing.assert_array_equal(model.types,[1,0,2,0])
    np.testing.assert_allclose(model.depths,.3)
    assert np.all(np.linalg.norm(model.pipes[:,3:]-model.pipes[:,:3],axis=1)>0)


def test_native_fluent_classes_execute_the_complete_owner_chain():
    pipes = CPipe.create(BOUNDARIES, FRACTURES)
    assert isinstance(pipes, CPipe)
    skeleton = pipes.Backbone()
    assert isinstance(skeleton, CBackbone)
    network = skeleton.Graph(aperture=lambda _length: 1e-3)
    assert isinstance(network, CGraph)
    solved = network.Solve([1, 0])
    assert solved.boundary_values.tolist() == [1, 0]
    assert all(np.isfinite(node.pressure) for node in solved.nodes)
