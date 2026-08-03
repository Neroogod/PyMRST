"""Native shale EDFM TPFA operators (`setupShaleEDFMOpsTPFA`)."""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.shale.data import PEDFMOperators, ShaleNNC, ShaleRock


def setup_shale_edfm_operators_tpfa(
    grid: Grid,
    rock: ShaleRock,
    face_transmissibility,
    nnc: ShaleNNC,
    *,
    cell_volumes=None,
) -> PEDFMOperators:
    """Assemble matrix faces and shale NNCs without pEDFM face reduction.

    This is the EDFM counterpart of ``setup_pedfm_operators_tpfa`` and ports
    MRST's shale-specific matrix mask, adsorption grain volume, pore volume,
    connection gradient/divergence, averaging and upwind operators.
    """
    t_all = np.asarray(face_transmissibility, dtype=float).reshape(-1)
    if t_all.size != grid.faces.num:
        raise ValueError("face_transmissibility must have one value per face")
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    matrix_neighbors = np.asarray(grid.faces.neighbors[internal], dtype=np.int64)
    neighbors = np.vstack([matrix_neighbors, nnc.cells])
    transmissibility = np.concatenate([t_all[internal], nnc.T])
    total_cells = max(
        np.asarray(rock.poro).size,
        int(neighbors.max()) + 1 if neighbors.size else grid.cells.num,
    )
    if np.asarray(rock.poro).size != total_cells:
        raise ValueError("rock properties must cover every global EDFM cell")

    columns = np.arange(neighbors.shape[0], dtype=np.int64)
    rows = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    cols = np.concatenate([columns, columns])
    signs = np.concatenate([-np.ones(columns.size), np.ones(columns.size)])
    divergence = sp.csr_array(
        (signs, (rows, cols)), shape=(total_cells, neighbors.shape[0])
    )

    if cell_volumes is None:
        volumes = np.concatenate([
            np.asarray(grid.cells.volumes, dtype=float),
            np.zeros(max(total_cells - grid.cells.num, 0)),
        ])
    else:
        volumes = np.asarray(cell_volumes, dtype=float).reshape(-1)
        if volumes.size != total_cells:
            raise ValueError("cell_volumes must cover every global EDFM cell")

    is_matrix = np.asarray(rock.is_matrix, dtype=bool).reshape(-1)
    if is_matrix.size != total_cells:
        raise ValueError("is_matrix must cover every global EDFM cell")
    is_sorbed = None
    grain_volume = None
    if "sorption" in rock.shale_mechanisms:
        is_sorbed = is_matrix.copy()
        grain_volume = is_sorbed * (1.0 - np.asarray(rock.poro)) * volumes

    return PEDFMOperators(
        neighbors=neighbors,
        transmissibility=transmissibility,
        transmissibility_all=t_all,
        face_multiplier=np.ones(grid.faces.num),
        divergence_matrix=divergence,
        is_matrix=is_matrix,
        is_sorbed=is_sorbed,
        grain_volume=grain_volume,
        pore_volume=np.asarray(rock.poro, dtype=float) * volumes,
    )


setupShaleEDFMOpsTPFA = setup_shale_edfm_operators_tpfa

__all__ = ["setupShaleEDFMOpsTPFA", "setup_shale_edfm_operators_tpfa"]
