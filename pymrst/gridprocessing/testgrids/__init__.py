"""MRST gridprocessing test-grid generators."""

from pymrst.gridprocessing.testgrids.cartesian_grdecl import cartesian_grdecl, cartesianGrdecl
from pymrst.gridprocessing.testgrids.create_pinched_column import create_pinched_column, createPinchedColumn
from pymrst.gridprocessing.testgrids.cut_cell_grids import cut_cell_grids, cutCellGrids
from pymrst.gridprocessing.testgrids.extruded_triangle_grid import extruded_triangle_grid, extrudedTriangleGrid
from pymrst.gridprocessing.testgrids.make_model3 import make_model3, makeModel3
from pymrst.gridprocessing.testgrids.mark_cut_grids_example1 import mark_cut_grids_example1, markCutGridsExample1
from pymrst.gridprocessing.testgrids.mark_cut_grids_example2 import mark_cut_grids_example2, markCutGridsExample2
from pymrst.gridprocessing.testgrids.one_sloping_fault import one_sloping_fault, oneSlopingFault
from pymrst.gridprocessing.testgrids.pinch_middle_cell import pinch_middle_cell, pinchMiddleCell
from pymrst.gridprocessing.testgrids.pinched_layers_grdecl import pinched_layers_grdecl, pinchedLayersGrdecl
from pymrst.gridprocessing.testgrids.pinched_node import pinched_node, pinchedNode
from pymrst.gridprocessing.testgrids.raised_column import raised_column, raisedColumn
from pymrst.gridprocessing.testgrids.simple_grdecl import simple_grdecl, simpleGrdecl
from pymrst.gridprocessing.testgrids.three_layers import three_layers, threeLayers
from pymrst.gridprocessing.testgrids.twister import twister

__all__ = [
    "cartesian_grdecl", "cartesianGrdecl",
    "create_pinched_column", "createPinchedColumn",
    "cut_cell_grids", "cutCellGrids",
    "extruded_triangle_grid", "extrudedTriangleGrid",
    "make_model3", "makeModel3",
    "mark_cut_grids_example1", "markCutGridsExample1",
    "mark_cut_grids_example2", "markCutGridsExample2",
    "one_sloping_fault", "oneSlopingFault",
    "pinch_middle_cell", "pinchMiddleCell",
    "pinched_layers_grdecl", "pinchedLayersGrdecl",
    "pinched_node", "pinchedNode",
    "raised_column", "raisedColumn",
    "simple_grdecl", "simpleGrdecl",
    "three_layers", "threeLayers", "twister",
]
