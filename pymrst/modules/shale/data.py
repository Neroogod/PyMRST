"""Shared data contracts for the MRST shale module port."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

from pymrst.params.rock import Rock
from pymrst.gridprocessing.grid import Grid
from pymrst.utils.adi import ad_matmul


@dataclass
class FracturePlane:
    """Polygonal hydraulic or natural fracture and its petrophysics."""

    points: np.ndarray
    aperture: float | None = None
    porosity: float | None = None
    permeability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError("fracture points must have shape (n, 3), n >= 3")
        self.points = points


@dataclass
class ShaleRock(Rock):
    """MRST ``makeShaleRock`` result with matrix and mechanism metadata."""

    is_matrix: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    shale_mechanisms: dict[str, Any] = field(default_factory=dict)

    @property
    def isMatrix(self) -> np.ndarray:  # noqa: N802 - MRST compatibility field
        return self.is_matrix

    @property
    def shaleMechanisms(self) -> Mapping[str, Any]:  # noqa: N802
        return self.shale_mechanisms


@dataclass
class ShaleNNC:
    """Global non-neighbor connections with MRST shale metadata."""

    cells: np.ndarray
    transmissibility: np.ndarray
    connection_type: tuple[str, ...]
    area: np.ndarray
    normal: np.ndarray
    plane_point: np.ndarray

    def __post_init__(self) -> None:
        self.cells = np.asarray(self.cells, dtype=np.int64).reshape(-1, 2)
        count = self.cells.shape[0]
        self.transmissibility = np.asarray(self.transmissibility, dtype=float).reshape(-1)
        self.area = np.asarray(self.area, dtype=float).reshape(-1)
        self.normal = np.asarray(self.normal, dtype=float).reshape(-1, 3)
        self.plane_point = np.asarray(self.plane_point, dtype=float).reshape(-1, 3)
        if not (
            self.transmissibility.size
            == self.area.size
            == self.normal.shape[0]
            == self.plane_point.shape[0]
            == len(self.connection_type)
            == count
        ):
            raise ValueError("all NNC fields must have one entry per connection")

    @property
    def T(self) -> np.ndarray:  # noqa: N802 - MRST field name
        return self.transmissibility

    @property
    def type(self) -> tuple[str, ...]:
        return self.connection_type

    @classmethod
    def empty(cls) -> "ShaleNNC":
        return cls(
            np.empty((0, 2), dtype=np.int64),
            np.empty(0),
            (),
            np.empty(0),
            np.empty((0, 3)),
            np.empty((0, 3)),
        )


@dataclass
class ShaleGlobalGrid:
    """Matrix grid plus ``hfm`` fracture cells and shale NNC metadata."""

    matrix: Grid
    fracture_volumes: np.ndarray
    fracture_centroids: np.ndarray
    fracture_plane: np.ndarray
    rock: ShaleRock
    nnc: ShaleNNC
    edfm: Any = None
    fractures: tuple[Any, ...] = ()
    projections: tuple[Any, ...] = ()
    grid: Grid | None = None
    fracture_cell_map: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    fracture_face_map: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    fracture_node_map: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    @property
    def n_matrix(self) -> int:
        return self.matrix.cells.num

    @property
    def n_fracture(self) -> int:
        return int(np.asarray(self.fracture_volumes).size)

    @property
    def n_cells(self) -> int:
        return self.n_matrix + self.n_fracture


@dataclass
class PEDFMOperators:
    """Connection operators produced by shale ``setupPEDFMOpsTPFA``."""

    neighbors: np.ndarray
    transmissibility: np.ndarray
    transmissibility_all: np.ndarray
    face_multiplier: np.ndarray
    divergence_matrix: sp.csr_array
    is_matrix: np.ndarray | None = None
    is_sorbed: np.ndarray | None = None
    grain_volume: np.ndarray | None = None
    pore_volume: np.ndarray | None = None

    @property
    def N(self) -> np.ndarray:  # noqa: N802
        return self.neighbors

    @property
    def T(self) -> np.ndarray:  # noqa: N802
        return self.transmissibility

    @property
    def T_all(self) -> np.ndarray:  # noqa: N802
        return self.transmissibility_all

    @property
    def C(self) -> sp.csr_array:  # noqa: N802
        return self.divergence_matrix.T

    @property
    def pv(self) -> np.ndarray | None:
        return self.pore_volume

    def grad(self, cell_values):
        """Connection gradient, including EDFM/pEDFM NNC rows."""
        return ad_matmul(self.C, cell_values)

    def avg(self, cell_values):
        """Arithmetic connection average with AD propagation."""
        count = self.neighbors.shape[0]
        rows = np.repeat(np.arange(count), 2)
        cols = self.neighbors.reshape(-1)
        matrix = sp.csr_array(
            (np.full(rows.size, 0.5), (rows, cols)),
            shape=(count, self.divergence_matrix.shape[0]),
        )
        return ad_matmul(matrix, cell_values)

    def upw(self, flag, cell_values):
        """Single-point upwind value on every matrix/NNC connection."""
        flag = np.asarray(flag, dtype=bool).reshape(-1)
        count = self.neighbors.shape[0]
        if flag.size != count:
            raise ValueError("upwind flag must have one value per connection")
        cols = np.where(flag, self.neighbors[:, 0], self.neighbors[:, 1])
        matrix = sp.csr_array(
            (np.ones(count), (np.arange(count), cols)),
            shape=(count, self.divergence_matrix.shape[0]),
        )
        return ad_matmul(matrix, cell_values)

    def div(self, flux) -> np.ndarray:
        return ad_matmul(self.divergence_matrix, flux)

    def acc_div(self, accumulation, flux) -> np.ndarray:
        return accumulation + self.div(flux)

    # MRST spellings used by equation owners.
    Div = div
    AccDiv = acc_div


@dataclass
class ProjectedNNC:
    """Projected matrix/fracture connections produced by pEDFM."""

    matrix_fracture_cells: np.ndarray
    matrix_matrix_neighbors: np.ndarray
    transmissibility: np.ndarray
    area: np.ndarray
    direction: np.ndarray
    connection_type: tuple[str, ...]

    def __post_init__(self) -> None:
        self.matrix_fracture_cells = np.asarray(
            self.matrix_fracture_cells, dtype=np.int64
        ).reshape(-1, 2)
        self.matrix_matrix_neighbors = np.asarray(
            self.matrix_matrix_neighbors, dtype=np.int64
        ).reshape(-1, 2)
        count = self.matrix_fracture_cells.shape[0]
        self.transmissibility = np.asarray(self.transmissibility, dtype=float).reshape(-1)
        self.area = np.asarray(self.area, dtype=float).reshape(-1)
        self.direction = np.asarray(self.direction, dtype=np.int64).reshape(-1)
        if not (
            self.matrix_matrix_neighbors.shape[0]
            == self.transmissibility.size
            == self.area.size
            == self.direction.size
            == len(self.connection_type)
            == count
        ):
            raise ValueError("all projected-NNC fields require equal row counts")

    @property
    def MFcells(self) -> np.ndarray:  # noqa: N802
        return self.matrix_fracture_cells

    @property
    def MMneighs(self) -> np.ndarray:  # noqa: N802
        return self.matrix_matrix_neighbors

    @property
    def T(self) -> np.ndarray:  # noqa: N802
        return self.transmissibility


@dataclass
class WellFractureConnections:
    """Intersections between piecewise-linear wells and fracture cells."""

    well_index: np.ndarray
    segment_index: np.ndarray
    fracture_index: np.ndarray
    points: np.ndarray
    matrix_cells: np.ndarray
    fracture_cells: np.ndarray

    def __post_init__(self) -> None:
        self.well_index = np.asarray(self.well_index, dtype=np.int64).reshape(-1)
        self.segment_index = np.asarray(self.segment_index, dtype=np.int64).reshape(-1)
        self.fracture_index = np.asarray(self.fracture_index, dtype=np.int64).reshape(-1)
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.matrix_cells = np.asarray(self.matrix_cells, dtype=np.int64).reshape(-1)
        self.fracture_cells = np.asarray(self.fracture_cells, dtype=np.int64).reshape(-1)
        count = self.well_index.size
        if not all(
            value == count
            for value in (
                self.segment_index.size,
                self.fracture_index.size,
                self.points.shape[0],
                self.matrix_cells.size,
                self.fracture_cells.size,
            )
        ):
            raise ValueError("well/fracture connection fields require equal lengths")


__all__ = [
    "FracturePlane",
    "PEDFMOperators",
    "ProjectedNNC",
    "ShaleGlobalGrid",
    "ShaleNNC",
    "ShaleRock",
    "WellFractureConnections",
]
