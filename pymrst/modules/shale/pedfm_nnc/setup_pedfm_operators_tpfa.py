"""pEDFM TPFA connection assembly (`setupPEDFMOpsTPFA`)."""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.shale.data import (
    PEDFMOperators,
    ProjectedNNC,
    ShaleNNC,
    ShaleRock,
)
from pymrst.modules.shale.pedfm_nnc.transmissibility_multiplier_pedfm import transmissibility_multiplier_pedfm


def setup_pedfm_operators_tpfa(
    grid: Grid,
    rock: ShaleRock,
    face_transmissibility,
    nnc: ShaleNNC,
    *,
    projected: ProjectedNNC | None = None,
    projected_matrix_neighbors=(),
    projected_areas=(),
    cell_volumes=None,
    tolerance: float = 1e-10,
) -> PEDFMOperators:
    """Assemble modified matrix-face and shale NNC connection operators."""
    t_all = np.asarray(face_transmissibility, dtype=float).reshape(-1)
    if t_all.size != grid.faces.num:
        raise ValueError("face_transmissibility must have one value per face")
    if projected is not None:
        if len(projected_matrix_neighbors) or len(projected_areas):
            raise ValueError(
                "pass projected or explicit projected arrays, not both"
            )
        projected_matrix_neighbors = projected.MMneighs
        projected_areas = projected.area
        nnc = ShaleNNC(
            cells=np.vstack([nnc.cells, projected.MFcells]),
            transmissibility=np.concatenate([nnc.T, projected.T]),
            connection_type=nnc.type + projected.connection_type,
            area=np.concatenate([nnc.area, projected.area]),
            normal=np.vstack([
                nnc.normal,
                np.zeros((projected.MFcells.shape[0], 3)),
            ]),
            plane_point=np.vstack([
                nnc.plane_point,
                np.zeros((projected.MFcells.shape[0], 3)),
            ]),
        )
    multiplier = transmissibility_multiplier_pedfm(
        grid,
        projected_matrix_neighbors,
        projected_areas,
        tolerance=tolerance,
    )
    t_all = t_all * multiplier
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    matrix_neighbors = np.asarray(grid.faces.neighbors[internal], dtype=np.int64)
    neighbors = np.vstack([matrix_neighbors, nnc.cells])
    transmissibility = np.concatenate([t_all[internal], nnc.T])
    total_cells = max(
        int(np.asarray(rock.poro).size),
        int(neighbors.max()) + 1 if neighbors.size else grid.cells.num,
    )
    columns = np.arange(neighbors.shape[0], dtype=np.int64)
    rows = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    cols = np.concatenate([columns, columns])
    values = np.concatenate([-np.ones(columns.size), np.ones(columns.size)])
    divergence = sp.csr_array(
        (values, (rows, cols)), shape=(total_cells, neighbors.shape[0])
    )

    is_matrix = np.asarray(rock.is_matrix, dtype=bool).reshape(-1)
    is_sorbed = None
    grain_volume = None
    if cell_volumes is None:
        volumes = np.concatenate([
            np.asarray(grid.cells.volumes, dtype=float),
            np.zeros(max(total_cells - grid.cells.num, 0), dtype=float),
        ])
    else:
        volumes = np.asarray(cell_volumes, dtype=float).reshape(-1)
        if volumes.size != total_cells:
            raise ValueError("cell_volumes must contain one value per global cell")
        if np.any(volumes <= 0):
            raise ValueError("cell_volumes must be positive")
    if "sorption" in rock.shale_mechanisms:
        if is_matrix.size != total_cells:
            raise ValueError("is_matrix must cover all global cells")
        is_sorbed = is_matrix.copy()
        grain_volume = is_sorbed * (1.0 - np.asarray(rock.poro)) * volumes
    return PEDFMOperators(
        neighbors=neighbors,
        transmissibility=transmissibility,
        transmissibility_all=t_all,
        face_multiplier=multiplier,
        divergence_matrix=divergence,
        is_matrix=is_matrix,
        is_sorbed=is_sorbed,
        grain_volume=grain_volume,
        pore_volume=np.asarray(rock.poro, dtype=float) * volumes,
    )


__all__ = ["setup_pedfm_operators_tpfa"]
