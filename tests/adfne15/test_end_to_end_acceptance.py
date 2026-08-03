"""Reference-scale, cross-owner ADFNE15 acceptance workflows."""
import numpy as np
import matplotlib

matplotlib.use("Agg")

from pymrst.modules.adfne15 import (backbone, contour, core3d_natural_fractures,
                                    demo, draw, export, graph, kriging,
                                    natural_fractures_2d_direction, pipe, solve,
                                    stereonet, variocloud, variogram)


def test_2d_network_to_flow_render_and_export(tmp_path):
    generated,total=natural_fractures_2d_direction(seed=11)
    assert len(generated.fractures)>100 and total>0
    boundaries=np.array([[0,0,0,1],[1,0,1,1]],float)
    fractures=np.array([[0,.5,1,.5],[.5,.5,.5,.8]],float)
    solved=solve(graph(backbone(pipe(boundaries,fractures)),
                       aperture=lambda _length:1e-3),[1,0])
    pressures=np.array([node.pressure for node in solved.nodes])
    assert np.all(np.isfinite(pressures)) and pressures.min()>=0 and pressures.max()<=1
    axis=draw('line',generated.fractures[:30])
    image=export(axis,tmp_path/'network.png',dpi=70)
    assert image.stat().st_size>1000
    svg=export(generated.fractures[:30],tmp_path/'network.svg')
    assert svg.read_text().count('<line')==30


def test_3d_cylindrical_domain_to_intersection_render_and_mesh_exports(tmp_path):
    result=core3d_natural_fractures()
    assert len(result.boundaries)==22 and len(result.fractures)>50
    vtk=export(result.fractures,tmp_path/'core.vtk')
    text=vtk.read_text(); assert 'DATASET POLYDATA' in text
    assert f'POLYGONS {len(result.fractures)} ' in text
    geo=export(result.fractures[:5],tmp_path/'core.geo')
    assert geo.read_text().count('Plane Surface')==5
    axis=draw('poly',result.fractures[:20],alpha=.25)
    png=export(axis,tmp_path/'core.png',dpi=70)
    assert png.stat().st_size>1000


def test_geostatistics_and_orientation_acceptance():
    rng=np.random.default_rng(8); points=rng.random((30,2)); values=np.sin(points[:,0]*3)+points[:,1]
    cloud=variocloud(points,values); experimental=variogram(cloud.distances,cloud.semivariances,cloud.variance,.5)
    assert experimental.counts.sum()>0
    estimate=kriging(points,values,model={'name':'exp','nugget':0,'sill':1,'range':1},grid_shape=(9,7))
    assert estimate.estimates.shape==(7,9) and np.all(np.isfinite(estimate.estimates))
    surface=contour(points,values,resolution=12); assert surface.values.shape==(12,12)
    orientation=stereonet(rng.uniform(0,90,100),rng.uniform(0,360,100),density=True)
    assert orientation.density.sum()==100
    assert demo(seed=5)['variogram_cloud'] is not None
