"""Qualification for ADFNE15 presentation, export and script owners."""
from datetime import datetime
import io
import numpy as np

from pymrst.modules.adfne15 import (CExt, Ticot, contour, demo, dfn_bin,
                                    dfn_orig, display, draw, export, grid,
                                    natural_fractures_2d_direction,
                                    natural_fractures_2d_random,
                                    omo_orientation_case, stereonet, summary,
                                    time_string)


def test_grid_cext_and_legacy_dfn_owners():
    np.testing.assert_allclose(grid(2), [[.5, 0, .5, 1], [0, .5, 1, .5]])
    definition = np.array([0,0,0, 1,0,0, 0,1,0], float)
    np.testing.assert_allclose(CExt.plane_normal(definition), [0,0,1])
    np.testing.assert_allclose(CExt.intersect_line_plane([[0,0,1,0,0,-2]], definition),
                               [[0,0,0]])
    np.testing.assert_allclose(CExt.point_on_plane([[1,2,3]], definition), [[1,2,0]])
    first = dfn_orig(dim=2, n=8, minl=.1, mu=.2, maxl=.3, seed=2)
    second = dfn_bin(dim=2, n=8, minl=.1, mu=.2, maxl=.3, seed=2)
    np.testing.assert_array_equal(first.fractures, second.fractures)


def test_contour_stereonet_and_draw_are_renderer_safe():
    points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    result=contour(points,[0,1,1,2],resolution=5)
    assert result.values.shape==(5,5) and np.all(np.isfinite(result.values))
    net=stereonet([0,45,90],[0,90,180],density=True,ndip=3,ndir=4)
    np.testing.assert_allclose(net.x,[0,0,-1],atol=1e-14)
    assert net.density.sum()==3
    import matplotlib
    matplotlib.use('Agg')
    axis=draw('line',[[0,0,1,1]])
    assert len(axis.lines)==1


def test_export_text_mesh_and_figure_formats(tmp_path):
    polygon=[np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])]
    for suffix in ('vtk','geo','txt','json','html'):
        path=export(polygon,tmp_path/f'network.{suffix}')
        assert path.exists() and path.stat().st_size>0
    svg=export([[0,0,1,1]],tmp_path/'network.svg')
    assert '<line' in svg.read_text()
    xyz=export(np.array([[1.,2.,3.]]),tmp_path/'points.xyz')
    assert xyz.exists()
    inp=export((polygon[0],np.array([[0,1,2]])),tmp_path/'mesh.inp')
    assert '*Element' in inp.read_text()


def test_reporting_timing_and_callable_examples():
    assert display('a',1)=='a 1'
    assert 'a' in summary([('a','b')])
    fixed=datetime(2024,1,2,3,4,5,678000)
    assert time_string(now=fixed)=='20240102030405'
    assert time_string(True,now=fixed)=='2024-01-02,03-04-05-678'
    timer=Ticot(); timer.start('work'); record=timer.stop(); assert record[0]=='work'
    directed,total=natural_fractures_2d_direction(count=5,seed=1)
    random,total_random=natural_fractures_2d_random(count=5,seed=1)
    assert len(directed.fractures)>0 and total>0 and len(random.fractures)>0 and total_random>0
    case=omo_orientation_case(); assert case.density.sum()==28
    demonstration=demo(seed=2); assert demonstration['info'].count>0
