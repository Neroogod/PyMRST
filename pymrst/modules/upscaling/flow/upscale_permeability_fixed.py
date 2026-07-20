"""Fixed-boundary permeability upscaling (MRST ``upscalePermeabilityFixed.m``)."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.incomp import incomp_tpfa, init_single_fluid
from pymrst.modules.upscaling.coarsegrid.generate_coarse_grid import CoarseGrid
from pymrst.modules.upscaling.upscale import _block_subgrid
from pymrst.params.rock import Rock, make_rock
from pymrst.params.wells_and_bc import add_bc
from pymrst.solvers.state import init_res_sol
from pymrst.solvers.trans import compute_trans


def upscale_perm_fixed(g: Grid, rock: Rock, cg: CoarseGrid, dp: float = 1.0) -> np.ndarray:
    """Flow-based diagonal permeability upscaling with sealed lateral sides."""
    nd = g.griddim
    fluid = init_single_fluid(1.0, 1.0)
    kk = np.zeros((cg.n_blocks, nd))

    for b, cells in enumerate(cg.blocks()):
        sub, order = _block_subgrid(g, cells)
        srock = make_rock(sub, rock.perm[order], rock.poro[order, None])
        t = compute_trans(sub, srock)

        bd = np.any(sub.faces.neighbors == -1, axis=1)
        fc = sub.faces.centroids
        for d in range(nd):
            lo = fc[:, d].min()
            hi = fc[:, d].max()
            inlet = np.flatnonzero(bd & (fc[:, d] < lo + 1e-9))
            outlet = np.flatnonzero(bd & (fc[:, d] > hi - 1e-9))
            bc = add_bc(None, inlet, "pressure", dp)
            bc = add_bc(bc, outlet, "pressure", 0.0)
            st = incomp_tpfa(init_res_sol(sub, 0.0), sub, t, fluid, bc=bc)

            flux = float(np.abs(st.flux[outlet]).sum())
            area = float(sub.faces.areas[outlet].sum())
            length = hi - lo
            kk[b, d] = (flux / area) / (dp / length)
    return kk


upscale_permeability_fixed = upscale_perm_fixed

__all__ = ["upscale_perm_fixed", "upscale_permeability_fixed"]
