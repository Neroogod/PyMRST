"""Native Python port of the ADFNE 1.5 discrete-fracture toolkit.

Copyright (c) 2018 Dr. Younes Fadakar Alghalandis. Distributed under the
ADFNE 1.5 three-clause license retained in this package.
"""

from .angle import angle
from .backbone import BackboneModel, backbone
from .boundary import boundary
from .config import ADFNEConfig, DEFAULT_CONFIG
from .dfn import DFNResult, generate_dfn
from .bbox import bounding_box
from .center import center
from .clip import clip_lines, clip_polygons
from .length import lengths
from .plane import plane
from .pointline import pointline
from .polynorm import polygon_normals
from .polyline import polyline
from .project import project
from .unique import unique_rows
from .orientation import Orientation, orientation
from .intersect import IntersectionResult, intersect
from .group import group
from .graph import GraphEdge, GraphModel, GraphNode, graph
from .isolated import isolated
from .mesh import ADFNEMeshUnavailableError, mesh
from .naned import naned
from .net import net
from .not_ import not_indices
from .pipe import PipeModel, pipe
from .flatri import flatri
from .filter import filter_ids
from .rand import random_values
from .scale import scale
from .cluster import cluster
from .idw import inverse_distance_weighting
from .occurrence import occurrence
from .percolate import percolate
from .snap import snap
from .sort import sort_points
from .split import SplitResult, split
from .solve import conductance, solve
from .stack import stack
from .reshape import reshape
from .distance import distance
from .field import field
from .label import label
from .ray import ray
from .relabel import relabel
from .rotate import rotate
from .tensor import ADFNETensorUnavailableError, tensor
from .translate import translate
from .wrap import wrap
from .geometry import SphereMesh, ellipse, ellipsoid, geometry, sphere
from .getshapepoly import get_shape_poly
from .inpoly import inpoly
from .polyclip import polyclip
from .cbackbone import CBackbone
from .cdata import CData
from .cgraph import CGraph
from .convert import (as_cell, as_struct, clock, convert, normalized,
                      polygon_triangles, to_2d, to_3d)
from .cpipe import CPipe
from .globals import ADFNEGlobals, GLOBALS
from .option import option
from .setstructfields import setstructfields
from .info import NetworkInfo, info
from .kriging import KrigingResult, kriging
from .markov import markov
from .stats import Statistics, stats
from .upscaling import upscaling
from .variocloud import VariogramCloud, variocloud
from .variogram import ExperimentalVariogram, variogram
from .variomodel import variomodel
from .axes import axes
from .cext import CExt
from .contour import ContourResult, contour
from .core3d_naturalfracs import Core3DResult, core3d_natural_fractures
from .demo import demo
from .dfn_bin import dfn_bin
from .dfn_orig import dfn_orig
from .display import display
from .draw import draw
from .export import export
from .grid import grid
from .naturalfrac2d_direction import natural_fractures_2d_direction
from .naturalfrac2d_random import natural_fractures_2d_random
from .omo_cases import omo_orientation_case
from .stereonet import StereonetResult, stereonet
from .summary import summary
from .syncax import syncax
from .ticot import Ticot
from .time import time_string

DFN = generate_dfn

__all__ = ["ADFNEConfig", "ADFNEGlobals", "DEFAULT_CONFIG", "GLOBALS",
           "DFN", "DFNResult", "Orientation",
           "BackboneModel", "GraphEdge", "GraphModel", "GraphNode",
           "ADFNEMeshUnavailableError", "ADFNETensorUnavailableError",
           "CBackbone", "CData", "CGraph", "CPipe", "ExperimentalVariogram",
           "IntersectionResult", "KrigingResult", "NetworkInfo", "PipeModel",
           "SphereMesh", "SplitResult", "Statistics", "VariogramCloud",
           "angle", "as_cell", "as_struct", "backbone", "boundary",
           "bounding_box", "center", "clock", "convert",
           "clip_lines", "clip_polygons", "conductance",
           "cluster", "distance", "field", "generate_dfn", "info",
           "inverse_distance_weighting", "kriging", "label", "lengths", "markov",
           "ellipse", "ellipsoid", "filter_ids", "flatri", "geometry", "get_shape_poly",
           "graph", "group", "inpoly", "intersect", "isolated",
           "mesh", "naned", "net", "normalized", "not_indices", "occurrence",
           "orientation", "pipe",
           "option", "percolate", "plane", "pointline", "polyclip",
           "polygon_triangles", "setstructfields", "snap",
           "sort_points", "sphere", "split", "stats",
           "polygon_normals", "polyline", "project", "random_values", "ray",
           "relabel", "reshape", "rotate", "scale", "solve", "stack",
           "tensor", "to_2d", "to_3d", "translate", "unique_rows",
           "upscaling", "variocloud", "variogram", "variomodel", "wrap"]

__all__ += ["CExt", "ContourResult", "Core3DResult", "StereonetResult",
            "Ticot", "axes", "contour", "core3d_natural_fractures", "demo",
            "dfn_bin", "dfn_orig", "display", "draw", "export", "grid",
            "natural_fractures_2d_direction", "natural_fractures_2d_random",
            "omo_orientation_case", "stereonet", "summary", "syncax",
            "time_string"]
